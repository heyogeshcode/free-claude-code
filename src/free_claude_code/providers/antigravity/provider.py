"""Antigravity provider implementation using Google Cloud Code / Antigravity API."""

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import httpx
from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native import NativeMessagesOptions
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    AnthropicToResponsesStream,
    MessagesReplayOrigin,
    OpenAIResponsesRequest,
    build_responses_messages_request,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.http import ProviderAttemptScope

from .auth import AntigravityAuthManager
from .models import (
    DEFAULT_ANTIGRAVITY_MODELS,
    DEFAULT_MODELS_HOSTS,
    parse_antigravity_models,
)
from .translator import AntigravityStreamTranslator, translate_messages_request


class AntigravityProvider(BaseProvider):
    """Serve Google Antigravity models (Gemini 3.8 Flash, Claude Sonnet 4.6, etc.) to Claude Code & OpenCode."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: AntigravityAuthManager,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._admission = admission
        self._signature_cache: dict[str, str] = {}
        self._client = client or httpx.AsyncClient(
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
        )
        self._owns_client = client is None
        self._closing = False

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        if not self._auth.is_connected():
            raise ExecutionFailure(
                FailureKind.AUTHENTICATION,
                401,
                "Google Antigravity account is not connected. Please log in via the FCC admin panel.",
                False,
            )
        if not request.model.strip():
            raise InvalidRequestError("Model name cannot be empty.")

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        if not self._auth.is_connected():
            raise ExecutionFailure(
                FailureKind.AUTHENTICATION,
                401,
                "Google Antigravity account is not connected. Please log in via the FCC admin panel.",
                False,
            )
        if not request.model.strip():
            raise InvalidRequestError("Model name cannot be empty.")

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Fetch live models from Antigravity backend with offline fallback."""
        if not self._auth.is_connected():
            return frozenset(DEFAULT_ANTIGRAVITY_MODELS)

        try:
            access = await self._auth.access()
        except Exception as exc:
            logger.debug("Antigravity access failed during model discovery: {}", exc)
            return frozenset(DEFAULT_ANTIGRAVITY_MODELS)

        for host in DEFAULT_MODELS_HOSTS:
            url = f"https://{host}/v1internal:fetchAvailableModels"
            try:
                response = await self._client.post(
                    url,
                    json={},
                    headers={
                        "Authorization": f"Bearer {access.access_token}",
                        "Content-Type": "application/json",
                        "User-Agent": "Antigravity",
                    },
                )
                if response.is_success:
                    data = response.json()
                    infos = parse_antigravity_models(data)
                    self._auth.set_model_count(len(infos))
                    return infos
            except Exception as exc:
                logger.debug("Failed to fetch models from {}: {}", host, exc)

        return frozenset(DEFAULT_ANTIGRAVITY_MODELS)

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream Anthropic SSE response from Antigravity streamGenerateContent."""
        self.preflight_messages(request, reasoning=reasoning)

        execution = self._admission.start_execution(request_id=request_id)
        public_model = response_model or request.model
        translator = AntigravityStreamTranslator(
            public_model,
            self._signature_cache,
            input_tokens=input_tokens,
        )

        while execution.can_attempt:
            scope: ProviderAttemptScope | None = None
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name="antigravity",
                    request_id=execution.request_id,
                )

                access = await self._auth.access()
                payload, _backend_model = translate_messages_request(
                    request,
                    access.project_id,
                    self._signature_cache,
                )

                headers = {
                    "Authorization": f"Bearer {access.access_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "Antigravity",
                }

                # Try primary host first, then secondary
                response: httpx.Response | None = None
                for host in DEFAULT_MODELS_HOSTS:
                    url = f"https://{host}/v1internal:streamGenerateContent?alt=sse"
                    try:
                        req = self._client.build_request(
                            "POST", url, json=payload, headers=headers
                        )
                        response = scope.retain(
                            await self._client.send(req, stream=True)
                        )
                        if response.status_code == 404:
                            await response.aclose()
                            response = None
                            continue
                        break
                    except httpx.ConnectError, httpx.TimeoutException:
                        if response is not None:
                            await response.aclose()
                        response = None
                        continue

                if response is None:
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        503,
                        "All Antigravity upstream endpoints failed to connect.",
                        True,
                    )

                if response.status_code == 401:
                    # Token might have been revoked; force refresh
                    await response.aclose()
                    access = await self._auth.access(force_refresh=True)
                    continue

                if response.status_code == 429:
                    detail = await response.aread()
                    await response.aclose()
                    raise ExecutionFailure(
                        FailureKind.RATE_LIMIT,
                        429,
                        f"Antigravity quota exhausted: {detail.decode('utf-8', errors='replace')[:200]}",
                        True,
                    )

                if not response.is_success:
                    err_body = await response.aread()
                    await response.aclose()
                    msg = err_body.decode("utf-8", errors="replace")[:300]
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        response.status_code,
                        f"Antigravity error ({response.status_code}): {msg}",
                        response.status_code >= 500,
                    )

                async for line in response.aiter_lines():
                    events = translator.process_line(line)
                    for ev in events:
                        yield ev

                for ev in translator.finalize():
                    yield ev

                execution.succeed()
                return

            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as exc:
                if not execution.can_attempt:
                    execution.fail(exc)
                    if isinstance(exc, ExecutionFailure):
                        raise
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        500,
                        f"Antigravity stream failed: {exc}",
                        False,
                    ) from exc

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Adapt Antigravity stream to OpenAI Responses format."""
        options = NativeMessagesOptions(
            model=request.model,
            max_tokens=request.max_output_tokens or 8192,
        )
        prepared = build_responses_messages_request(
            request,
            options=options,
            replay_scope="antigravity",
        )
        messages_req = MessagesRequest.model_validate(prepared.body)
        presenter = AnthropicToResponsesStream(
            request,
            public_model=response_model or request.model,
            tool_identities=prepared.tool_identities,
            replay_origin=MessagesReplayOrigin("antigravity", request.model),
        )
        for frame in presenter.start():
            yield frame

        decoder = AnthropicSSEDecoder()
        async for sse_chunk in self.stream_messages(
            messages_req,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        ):
            for event in decoder.feed(sse_chunk):
                payload = cast(JsonObject, event.data)
                kind = event.event or payload.get("type")
                if isinstance(kind, str) and kind:
                    for frame in presenter.feed(kind, payload):
                        yield frame
        for event in decoder.finish():
            payload = cast(JsonObject, event.data)
            kind = event.event or payload.get("type")
            if isinstance(kind, str) and kind:
                for frame in presenter.feed(kind, payload):
                    yield frame

    async def cleanup(self) -> None:
        self._closing = True
        if self._owns_client:
            await self._client.aclose()
