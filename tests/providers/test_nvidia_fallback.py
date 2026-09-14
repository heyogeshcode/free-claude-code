"""Unit tests for NvidiaFallbackProvider."""

from unittest.mock import AsyncMock

import pytest

from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.nvidia_fallback.client import NvidiaFallbackProvider


def _dummy_provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="nvapi-dummy",
        base_url="https://integrate.api.nvidia.com/v1",
        rate_limit=100,
        rate_window=60,
        max_concurrency=10,
        http_read_timeout=30.0,
        http_write_timeout=10.0,
        http_connect_timeout=5.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )


def test_nvidia_fallback_preflight():
    provider = NvidiaFallbackProvider(_dummy_provider_config())

    # Empty messages should fail preflight
    with pytest.raises(ExecutionFailure) as exc:
        provider.preflight_messages(
            MessagesRequest(model="meta/llama-3.3-70b-instruct", messages=[])
        )
    assert exc.value.status_code == 400

    # Valid messages should pass preflight
    provider.preflight_messages(
        MessagesRequest(
            model="meta/llama-3.3-70b-instruct",
            messages=[Message(role="user", content="hello")],
        )
    )


@pytest.mark.asyncio
async def test_nvidia_fallback_stream_messages():
    mock_hedging = AsyncMock()

    async def mock_stream(*args, **kwargs):
        yield "event: message_start\ndata: {}\n\n"
        yield "event: content_block_delta\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    mock_hedging.stream_chat_completions = mock_stream

    provider = NvidiaFallbackProvider(
        _dummy_provider_config(), hedging_engine=mock_hedging
    )
    req = MessagesRequest(
        model="meta/llama-3.3-70b-instruct",
        messages=[Message(role="user", content="hello")],
    )

    chunks = [chunk async for chunk in provider.stream_messages(req)]

    assert len(chunks) == 3
    assert "event: message_start" in chunks[0]


@pytest.mark.asyncio
async def test_nvidia_fallback_list_models(monkeypatch):
    import httpx

    async def mock_fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_fail)

    provider = NvidiaFallbackProvider(_dummy_provider_config())
    models = await provider.list_model_infos()
    assert len(models) >= 2
    model_ids = {m.model_id for m in models}
    assert "meta/llama-3.3-70b-instruct" in model_ids


@pytest.mark.asyncio
async def test_nvidia_fallback_list_models_upstream(monkeypatch):
    import httpx

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {"id": "meta/llama-3.3-70b-instruct"},
                    {"id": "deepseek-ai/deepseek-r1"},
                    {"id": "mistralai/mistral-large-2-instruct"},
                    {"id": "nvidia/nemotron-4-340b-instruct"},
                ]
            }

    async def mock_get(self, url, headers=None, **kwargs):
        assert "models" in url
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    mock_hedging = AsyncMock()
    from unittest.mock import MagicMock
    mock_ks = MagicMock()
    mock_ks.get_healthy_keys.return_value = ["nvapi-mock-key"]
    mock_hedging._key_store = mock_ks

    provider = NvidiaFallbackProvider(
        _dummy_provider_config(), hedging_engine=mock_hedging
    )
    models = await provider.list_model_infos()
    assert len(models) == 4
    model_ids = {m.model_id for m in models}
    assert "nvidia/nemotron-4-340b-instruct" in model_ids
    assert "mistralai/mistral-large-2-instruct" in model_ids
