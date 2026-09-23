"""Antigravity provider implementation using Google Cloud Code / Antigravity API."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from collections import OrderedDict
from typing import Any, cast

import httpx
from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.paths import antigravity_signatures_path
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
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose

from .auth import AntigravityAuthManager
from .models import (
    DEFAULT_ANTIGRAVITY_MODELS,
    DEFAULT_MODELS_HOSTS,
    parse_antigravity_models,
)
from .translator import AntigravityStreamTranslator, translate_messages_request

class SignatureCache:
    def __init__(self, max_size: int = 5000):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size
    
    def get(self, key: str, default: Any = None) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return default
    
    def set(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            
    def __setitem__(self, key: str, value: str) -> None:
        self.set(key, value)
        
    def __getitem__(self, key: str) -> str:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
    
    def items(self) -> Any:
        return self._cache.items()
    
    def update(self, data: dict) -> None:
        for k, v in data.items():
            self.set(k, v)

    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]


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
        self._signatures_path = antigravity_signatures_path()
        self._signature_cache: SignatureCache = self._load_signature_cache()
        try:
            import h2  # noqa: F401
            has_h2 = True
        except ImportError:
            has_h2 = False

        self._client = client or httpx.AsyncClient(
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30),
            http2=has_h2,
        )
        self._owns_client = client is None
        self._closing = False

    def _load_signature_cache(self) -> SignatureCache:
        cache = SignatureCache(max_size=5000)
        if not self._signatures_path.exists():
            return cache
        try:
            with open(self._signatures_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    cache.update({str(k): str(v) for k, v in data.items()})
        except Exception as exc:
            logger.debug("Failed to load Antigravity signature cache: {}", exc)
        return cache

    def _save_signature_cache(self) -> None:
        try:
            self._signatures_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._signatures_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._signature_cache.items()), f)
            temp_path.replace(self._signatures_path)
        except Exception as exc:
            logger.debug("Failed to save Antigravity signature cache: {}", exc)

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

        sanitized = _sanitize_responses_request(request)
        options = NativeMessagesOptions(
            model=sanitized.model,
            max_tokens=sanitized.max_output_tokens or 8192,
        )
        build_responses_messages_request(
            sanitized,
            options=options,
            replay_scope="antigravity",
        )

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

    def stream_messages(
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
        return self._stream_messages(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )

    async def _stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(request_id=request_id)
        run = self._run_messages(
            request,
            input_tokens=input_tokens,
            execution=execution,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            async for event in run:
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            execution.succeed()
        finally:
            await maybe_await_aclose(run)
            execution.abandon()

    async def _run_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        execution: ProviderExecution,
        response_model: str | None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
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
                saw_404: bool = False
                last_404_body: str = ""
                for host in DEFAULT_MODELS_HOSTS:
                    url = f"https://{host}/v1internal:streamGenerateContent?alt=sse"
                    try:
                        req = self._client.build_request(
                            "POST", url, json=payload, headers=headers
                        )
                        res = await self._client.send(req, stream=True)
                        if res.status_code == 404:
                            saw_404 = True
                            body = await res.aread()
                            last_404_body = body.decode("utf-8", errors="replace")[:200]
                            await res.aclose()
                            continue
                        response = res
                        break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        continue

                if response is None:
                    if saw_404:
                        raise ExecutionFailure(
                            FailureKind.UPSTREAM,
                            404,
                            f"Model '{request.model}' (backend '{_backend_model}') not found on Antigravity: {last_404_body}",
                            False,
                        )
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        503,
                        "All Antigravity upstream endpoints failed to connect.",
                        True,
                    )

                if response.status_code == 401:
                    # Token might have been revoked; force refresh
                    await response.aclose()
                    await self._auth.access(force_refresh=True)
                    await scope.aclose(active_error=None)
                    scope = None
                    continue

                if response.status_code == 429:
                    if saw_404:
                        await response.aclose()
                        raise ExecutionFailure(
                            FailureKind.UPSTREAM,
                            404,
                            f"Model '{request.model}' (backend '{_backend_model}') not found on Antigravity: {last_404_body}",
                            False,
                        )
                    detail = await response.aread()
                    retry_after = response.headers.get("Retry-After")
                    await response.aclose()
                    msg = f"Antigravity quota exhausted: {detail.decode('utf-8', errors='replace')[:200]}"
                    if retry_after:
                        msg += f" (Retry-After: {retry_after}s)"
                    raise ExecutionFailure(
                        FailureKind.RATE_LIMIT,
                        429,
                        msg,
                        True,
                    )

                if not response.is_success:
                    err_body = await response.aread()
                    await response.aclose()
                    msg = err_body.decode("utf-8", errors="replace")
                    
                    if response.status_code == 400:
                        try:
                            import re
                            err_json = json.loads(msg)
                            err_msg = err_json.get("error", {}).get("message", "")
                            if "thoughtSignature" in err_msg or "signature mismatch" in err_msg.lower():
                                logger.warning("Antigravity signature mismatch detected: {}", err_msg)
                                match = re.search(r"tool[ _]call['\s]+([a-zA-Z0-9_-]+)", err_msg)
                                if match:
                                    call_id = match.group(1)
                                    logger.warning("Clearing signature cache for tool call: {}", call_id)
                                    self._signature_cache.delete(call_id)
                                else:
                                    logger.warning("Clearing entire signature cache due to unparseable tool call ID")
                                    self._signature_cache._cache.clear()
                        except Exception as e:
                            logger.debug("Failed to parse 400 error body for signature mismatch: {}", e)

                    msg_trunc = msg[:300]
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        response.status_code,
                        f"Antigravity error ({response.status_code}): {msg_trunc}",
                        response.status_code >= 500,
                    )

                scope.retain(response)

                import asyncio
                
                q = asyncio.Queue()
                
                async def read_stream():
                    try:
                        async for chunk in response.aiter_bytes():
                            await q.put(("chunk", chunk))
                        await q.put(("done", None))
                    except Exception as e:
                        await q.put(("error", e))
                        
                reader_task = asyncio.create_task(read_stream())
                
                buffer = b""
                is_thinking = False
                
                try:
                    while True:
                        try:
                            msg_type, data = await asyncio.wait_for(q.get(), timeout=5.0)
                            if msg_type == "done":
                                break
                            elif msg_type == "error":
                                raise data
                            elif msg_type == "chunk":
                                buffer += data
                                while b"\n" in buffer:
                                    line_bytes, buffer = buffer.split(b"\n", 1)
                                    try:
                                        line = line_bytes.decode("utf-8")
                                        events = translator.process_line(line)
                                        if events and not attempt.accepted:
                                            await attempt.accept()
                                        for ev in events:
                                            if '"type":"thinking"' in ev or '"type": "thinking"' in ev:
                                                is_thinking = True
                                            elif '"type":"text"' in ev or '"type": "text"' in ev:
                                                is_thinking = False
                                            yield ev
                                    except Exception as e:
                                        logger.debug("Skipping malformed SSE line: {}", e)
                        except asyncio.TimeoutError:
                            if is_thinking:
                                yield ": heartbeat\n\n"
                finally:
                    reader_task.cancel()

                if buffer:
                    try:
                        line = buffer.decode("utf-8", errors="replace")
                        events = translator.process_line(line)
                        if events and not attempt.accepted:
                            await attempt.accept()
                        for ev in events:
                            yield ev
                    except Exception as e:
                        logger.debug("Skipping malformed SSE line: {}", e)

                for ev in translator.finalize():
                    if not attempt.accepted:
                        await attempt.accept()
                    yield ev

                self._save_signature_cache()
                return

            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as exc:
                if not execution.can_attempt:
                    if isinstance(exc, ExecutionFailure):
                        raise
                    raise ExecutionFailure(
                        FailureKind.UPSTREAM,
                        500,
                        f"Antigravity stream failed: {exc}",
                        False,
                    ) from exc
            finally:
                self._save_signature_cache()
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

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
        sanitized = _sanitize_responses_request(request)
        options = NativeMessagesOptions(
            model=sanitized.model,
            max_tokens=sanitized.max_output_tokens or 8192,
        )
        prepared = build_responses_messages_request(
            sanitized,
            options=options,
            replay_scope="antigravity",
        )
        messages_req = MessagesRequest.model_validate(prepared.body)
        presenter = AnthropicToResponsesStream(
            sanitized,
            public_model=response_model or sanitized.model,
            tool_identities=prepared.tool_identities,
            replay_origin=MessagesReplayOrigin("antigravity", sanitized.model),
        )
        for frame in presenter.start():
            yield frame

        decoder = AnthropicSSEDecoder()
        stream = self.stream_messages(
            messages_req,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            async for sse_chunk in stream:
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
        finally:
            await maybe_await_aclose(stream)

    async def cleanup(self) -> None:
        self._closing = True
        if self._owns_client:
            await self._client.aclose()


_ALLOWED_REQUEST_FIELDS = {
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stream",
    "temperature",
    "top_p",
    "max_output_tokens",
    "metadata",
    "reasoning",
    "previous_response_id",
    "store",
    "text",
    "include",
    "truncation",
}


def _sanitize_responses_request(
    request: OpenAIResponsesRequest,
) -> OpenAIResponsesRequest:
    """Sanitize OpenAI Responses request so it strictly satisfies Messages translation requirements.

    Codex and other clients pass fields like truncation='auto', reasoning.summary='none',
    text.verbosity, prompt_cache_key, client_metadata, and non-function tool definitions
    (e.g. web_search) which are rejected by strict Messages wire shape converters.
    """
    raw_data = request.model_dump(mode="json", exclude_none=True)
    # Strip unknown root-level fields like prompt_cache_key, client_metadata
    data = {k: v for k, v in raw_data.items() if k in _ALLOWED_REQUEST_FIELDS}

    # 1. Truncation: only None or 'disabled' allowed by Messages converter
    if "truncation" in data and data["truncation"] not in (None, "disabled"):
        data.pop("truncation", None)

    # 2. Include: only 'reasoning.encrypted_content' allowed
    if "include" in data:
        include = data["include"]
        if isinstance(include, list):
            filtered = [x for x in include if x == "reasoning.encrypted_content"]
            if filtered:
                data["include"] = filtered
            else:
                data.pop("include", None)
        else:
            data.pop("include", None)

    # 3. Reasoning: keep only effort and summary='auto'
    if "reasoning" in data:
        reasoning = data["reasoning"]
        if isinstance(reasoning, dict):
            clean_reasoning: dict[str, Any] = {}
            if "effort" in reasoning and reasoning["effort"] is not None:
                clean_reasoning["effort"] = reasoning["effort"]
            if reasoning.get("summary") == "auto":
                clean_reasoning["summary"] = "auto"
            if clean_reasoning:
                data["reasoning"] = clean_reasoning
            else:
                data.pop("reasoning", None)
        else:
            data.pop("reasoning", None)

    # 4. Text: only format allowed
    if "text" in data:
        text_val = data["text"]
        if isinstance(text_val, dict) and "format" in text_val:
            data["text"] = {"format": text_val["format"]}
        else:
            data.pop("text", None)

    # 5. Tools: only keep function, custom, and namespace tools
    valid_tool_names: set[str] = set()
    if "tools" in data:
        tools_val = data["tools"]
        if isinstance(tools_val, list):
            clean_tools: list[dict[str, Any]] = []
            for t in tools_val:
                if isinstance(t, dict) and t.get("type") in ("function", "custom", "namespace"):
                    clean_tools.append(t)
                    if "name" in t and isinstance(t["name"], str):
                        valid_tool_names.add(t["name"])
                    if t.get("type") == "namespace" and isinstance(t.get("tools"), list):
                        ns_name = t.get("name", "")
                        for sub_t in t["tools"]:
                            if isinstance(sub_t, dict) and "name" in sub_t and isinstance(sub_t["name"], str):
                                sub_name = sub_t["name"]
                                valid_tool_names.add(sub_name)
                                if ns_name:
                                    valid_tool_names.add(f"{ns_name}.{sub_name}")
                                    valid_tool_names.add(f"{ns_name}__{sub_name}")
            if clean_tools:
                data["tools"] = clean_tools
            else:
                data.pop("tools", None)
        else:
            data.pop("tools", None)

    # 6. Tool choice: ensure valid reference or auto
    if "tool_choice" in data:
        choice = data["tool_choice"]
        if not data.get("tools"):
            if isinstance(choice, dict) or choice in ("required", "any"):
                data.pop("tool_choice", None)
        elif isinstance(choice, dict) and choice.get("name") not in valid_tool_names:
            data["tool_choice"] = "auto"

    # 7. Input: sanitize message and content blocks
    if "input" in data and isinstance(data["input"], list):
        clean_input: list[Any] = []
        for item in data["input"]:
            if isinstance(item, str):
                clean_input.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in (None, "message"):
                    cleaned_item = {
                        k: v
                        for k, v in item.items()
                        if k in ("type", "role", "content", "id", "status")
                    }
                    content = cleaned_item.get("content")
                    if isinstance(content, list):
                        clean_blocks: list[Any] = []
                        for block in content:
                            if isinstance(block, dict):
                                b_type = block.get("type")
                                if b_type in ("input_text", "output_text", "text"):
                                    clean_blocks.append(
                                        {
                                            "type": "text"
                                            if b_type == "text"
                                            else b_type,
                                            "text": block.get("text", ""),
                                        }
                                    )
                                elif b_type == "input_image":
                                    clean_blocks.append(block)
                                else:
                                    if "text" in block:
                                        clean_blocks.append(
                                            {"type": "text", "text": str(block["text"])}
                                        )
                            elif isinstance(block, str):
                                clean_blocks.append({"type": "text", "text": block})
                        cleaned_item["content"] = clean_blocks
                    clean_input.append(cleaned_item)
                else:
                    clean_input.append(item)
            else:
                clean_input.append(item)
        data["input"] = clean_input

    return OpenAIResponsesRequest.model_validate(data)
