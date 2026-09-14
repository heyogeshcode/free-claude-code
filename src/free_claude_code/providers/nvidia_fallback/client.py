"""NVIDIA Fallback Hedging Provider."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.nvidia_hedging import (
    NvidiaHedgingEngine,
    get_nvidia_hedging_engine,
)
from free_claude_code.core.nvidia_normalizer import NvidiaSchemaNormalizer
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.model_listing import extract_openai_model_infos


class NvidiaFallbackProvider(BaseProvider):
    """NVIDIA Fallback provider using speculative parallel racing across multi-key pool."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController | None = None,
        hedging_engine: NvidiaHedgingEngine | None = None,
    ) -> None:
        super().__init__(config)
        self._admission = admission
        self._hedging = hedging_engine or get_nvidia_hedging_engine()

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        if not request.messages:
            raise ExecutionFailure(
                kind=FailureKind.INVALID_REQUEST,
                status_code=400,
                message="request must include at least one message",
                retryable=False,
            )

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        if request.input is None:
            raise ExecutionFailure(
                kind=FailureKind.INVALID_REQUEST,
                status_code=400,
                message="request must include input",
                retryable=False,
            )

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        target_model = response_model or request.model
        body = NvidiaSchemaNormalizer.anthropic_to_openai_chat(
            request,
            stream=True,
            default_model=target_model,
        )
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

        async for chunk in self._hedging.stream_chat_completions(
            body,
            model=target_model,
            message_id=msg_id,
            is_anthropic_wire=True,
        ):
            yield chunk

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        target_model = response_model or request.model
        # Responses input can be string or list
        input_data = request.input
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            messages = []

        body: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": True,
        }
        resp_id = f"resp_{uuid.uuid4().hex[:12]}"

        async for chunk in self._hedging.stream_chat_completions(
            body,
            model=target_model,
            message_id=resp_id,
            is_anthropic_wire=False,
        ):
            yield chunk

    async def cleanup(self) -> None:
        pass

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        key = None
        key_store = getattr(self._hedging, "_key_store", None)
        if key_store is not None:
            healthy = key_store.get_healthy_keys()
            if healthy:
                key = healthy[0]
            else:
                all_keys = key_store.get_all_keys()
                if all_keys:
                    key = all_keys[0]

        if not key and getattr(self._config, "api_key", None):
            cfg_key = self._config.api_key
            if cfg_key and cfg_key != "nvidia-pool":
                key = cfg_key

        if key:
            try:
                base_url = (self._config.base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")
                url = f"{base_url}/models"
                headers = {"Authorization": f"Bearer {key}"}
                proxy = getattr(self._config, "proxy", None) or None
                async with httpx.AsyncClient(proxy=proxy, timeout=10.0) as client:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        payload = resp.json()
                        infos = extract_openai_model_infos(
                            payload,
                            provider_name="NVIDIA Fallback Pool",
                        )
                        if infos:
                            return infos
            except Exception as exc:
                logger.warning(
                    f"NvidiaFallbackProvider: failed to fetch dynamic models from upstream: {exc}"
                )

        return frozenset(
            [
                ProviderModelInfo(
                    model_id="meta/llama-3.3-70b-instruct",
                    context_window_tokens=131072,
                    max_output_tokens=4096,
                ),
                ProviderModelInfo(
                    model_id="deepseek-ai/deepseek-r1",
                    supports_thinking=True,
                    context_window_tokens=131072,
                    max_output_tokens=8192,
                ),
            ]
        )
