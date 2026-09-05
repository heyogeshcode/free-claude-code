"""Tests for AntigravityProvider."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.antigravity.auth import (
    AntigravityAccess,
    AntigravityAuthManager,
)
from free_claude_code.providers.antigravity.provider import AntigravityProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, make_provider_config


@pytest.fixture
def provider_config():
    return make_provider_config(
        api_key=None,
        base_url="https://daily-cloudcode-pa.googleapis.com",
    )


@pytest.mark.asyncio
async def test_preflight_fails_when_disconnected(provider_config):
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = False

    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=immediate_admission(),
    )
    try:
        req = make_messages_request("antigravity/gemini-3.8-flash")
        with pytest.raises(ExecutionFailure) as exc_info:
            provider.preflight_messages(req)
        assert exc_info.value.status_code == 401
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_preflight_fails_on_empty_model(provider_config):
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True

    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=immediate_admission(),
    )
    try:
        req = make_messages_request("")
        with pytest.raises(InvalidRequestError):
            provider.preflight_messages(req)
    finally:
        await provider.cleanup()


@pytest.mark.asyncio
async def test_list_model_infos_fetches_upstream(provider_config):
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("mock_token", "mock_project")
    )
    auth.set_model_count = MagicMock()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer mock_token"
        assert request.headers.get("User-Agent") == "Antigravity"
        return httpx.Response(
            200,
            json={
                "models": {
                    "gemini-3.8-flash-tiered": {
                        "supportsThinking": True,
                        "supportsImages": True,
                    }
                }
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=immediate_admission(),
        client=client,
    )
    try:
        models = await provider.list_model_infos()
        model_ids = {m.model_id for m in models}
        assert "gemini-3.8-flash-tiered" in model_ids
        assert "gemini-3.8-flash" in model_ids
        auth.set_model_count.assert_called_once()
    finally:
        await provider.cleanup()
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_messages_success(provider_config):
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("ya29.test_token", "test_proj")
    )

    sse_body = (
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello from Antigravity!"}]}}]}}\n\n'
        b'data: {"response": {"candidates": [{"finishReason": "STOP"}]}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer ya29.test_token"
        assert request.headers.get("User-Agent") == "Antigravity"
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=immediate_admission(),
        client=client,
    )
    try:
        req = make_messages_request("antigravity/gemini-3.8-flash")
        chunks = [chunk async for chunk in provider.stream_messages(req)]
        joined = "".join(chunks)
        assert "event: message_start" in joined
        assert "Hello from Antigravity!" in joined
        assert "event: message_stop" in joined
    finally:
        await provider.cleanup()
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_responses_adaptation(provider_config):
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("ya29.test_token", "test_proj")
    )

    sse_body = (
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "OpenCode response"}]}}]}}\n\n'
        b'data: {"response": {"candidates": [{"finishReason": "STOP"}]}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"Content-Type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=immediate_admission(),
        client=client,
    )
    try:
        req = OpenAIResponsesRequest(
            model="antigravity/gemini-3.8-flash",
            input=[{"role": "user", "content": "Hello"}],
        )
        frames = [frame async for frame in provider.stream_responses(req)]
        joined = "".join(frames)
        assert "response.created" in joined or "response.output_item.added" in joined
    finally:
        await provider.cleanup()
        await client.aclose()
