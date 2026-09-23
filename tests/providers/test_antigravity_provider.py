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


@pytest.mark.asyncio
async def test_stream_messages_concurrency_release_multiple_requests(provider_config):
    """Verify that multiple consecutive requests release semaphore permits and do not deadlock."""
    from free_claude_code.providers.admission import ProviderAdmissionController

    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("ya29.test_token", "test_proj")
    )

    sse_body = (
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}]}}\n\n'
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
    # Strict max_concurrency=2: would deadlock on 3rd request if permits leaked
    admission = ProviderAdmissionController(
        provider_name="antigravity",
        rate_limit=100,
        rate_window=60,
        max_concurrency=2,
    )
    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=admission,
        client=client,
    )
    try:
        req = make_messages_request("antigravity/gemini-3.8-flash")
        for _ in range(5):
            chunks = [chunk async for chunk in provider.stream_messages(req)]
            joined = "".join(chunks)
            assert "Hello!" in joined
    finally:
        await provider.cleanup()
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_messages_early_exit_releases_permit(provider_config):
    """Verify that breaking early from a stream properly closes the scope and releases permit."""
    from free_claude_code.providers.admission import ProviderAdmissionController

    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("ya29.test_token", "test_proj")
    )

    sse_body = (
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Part 1"}]}}]}}\n\n'
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Part 2"}]}}]}}\n\n'
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
    admission = ProviderAdmissionController(
        provider_name="antigravity",
        rate_limit=100,
        rate_window=60,
        max_concurrency=1,  # Only 1 slot!
    )
    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=admission,
        client=client,
    )
    try:
        req = make_messages_request("antigravity/gemini-3.8-flash")
        # Request 1: break early after first chunk
        async for _ in provider.stream_messages(req):
            break

        # Request 2: should succeed immediately if permit was released
        chunks = [chunk async for chunk in provider.stream_messages(req)]
        assert len(chunks) > 0
    finally:
        await provider.cleanup()
        await client.aclose()


@pytest.mark.asyncio
async def test_codex_stream_responses_sanitization(provider_config):
    """Verify that Codex requests with truncation, web_search, summary=none, text.verbosity succeed."""
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("ya29.test_token", "test_proj")
    )

    sse_body = (
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Codex result"}]}}]}}\n\n'
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
        codex_req = OpenAIResponsesRequest(
            model="antigravity/gemini-3.8-flash",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the weather?"}],
                }
            ],
            tools=[
                {"type": "web_search"},
                {
                    "type": "namespace",
                    "name": "agent",
                    "description": "Agent namespace tool",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "description": "Spawn agent",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                },
                {
                    "type": "function",
                    "name": "bash",
                    "description": "Run bash command",
                    "parameters": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "cmd": {
                                "type": "string",
                                "propertyNames": {"pattern": "^[a-z]+$"},
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            ],
            tool_choice="auto",
            parallel_tool_calls=True,
            reasoning={"effort": "medium", "summary": "none"},
            truncation="auto",
            text={"verbosity": "low"},
            prompt_cache_key="01a071c0-69b8-7b43-b113-54fa6554812d",
            client_metadata={"session_id": "test-session-123"},
        )

        # Preflight should succeed
        provider.preflight_responses(codex_req)

        # Streaming should succeed without ResponsesConversionError
        frames = [frame async for frame in provider.stream_responses(codex_req)]
        joined = "".join(frames)
        assert "response.completed" in joined or "response.output_item.added" in joined
    finally:
        await provider.cleanup()
        await client.aclose()


def test_resolve_backend_model_aliases():
    from free_claude_code.providers.antigravity.models import resolve_backend_model

    assert resolve_backend_model("opus-4.6") == "claude-opus-4-6-thinking"
    assert resolve_backend_model("sonnet-4.6") == "claude-sonnet-4-6"
    assert resolve_backend_model("antigravity/opus-4.6") == "claude-opus-4-6-thinking"
    assert resolve_backend_model("antigravity/sonnet-4.6") == "claude-sonnet-4-6"
    assert resolve_backend_model("claude-opus-4.6") == "claude-opus-4-6-thinking"
    assert resolve_backend_model("claude-sonnet-4.6") == "claude-sonnet-4-6"
    assert resolve_backend_model("claude-3-7-sonnet-latest") == "claude-sonnet-4-6"
    assert resolve_backend_model("claude-3-5-sonnet") == "claude-sonnet-4-6"
    assert resolve_backend_model("claude-3-opus") == "claude-opus-4-6-thinking"
    assert resolve_backend_model("haiku") == "gemini-2.5-flash"
    assert resolve_backend_model("gemini-3.8-flash") == "gemini-3.8-flash-tiered"


def test_stream_translator_thinking_blocks():
    import json
    from free_claude_code.providers.antigravity.translator import AntigravityStreamTranslator

    cache = {}
    translator = AntigravityStreamTranslator(
        public_model="opus-4.6",
        signature_cache=cache,
        input_tokens=10,
    )

    # 1. Thought chunk with thought: True and text
    line1 = 'data: {"response": {"candidates": [{"content": {"parts": [{"thought": true, "text": "Let me think...", "thoughtSignature": "sig123"}]}}]}}\n'
    events1 = translator.process_line(line1)
    joined1 = "".join(events1)
    assert "message_start" in joined1
    assert "content_block_start" in joined1
    assert '"type": "thinking"' in joined1
    assert "thinking_delta" in joined1
    assert "signature_delta" in joined1
    assert cache.get("latest_turn_sig") == "sig123"

    # 2. Transition from thought to final text
    line2 = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello world!"}]}}]}}\n'
    events2 = translator.process_line(line2)
    joined2 = "".join(events2)
    assert "content_block_stop" in joined2
    assert '"type": "text"' in joined2
    assert "text_delta" in joined2

    # 3. Finalize
    events3 = translator.finalize()
    joined3 = "".join(events3)
    assert "content_block_stop" in joined3
    assert "message_delta" in joined3
    assert "message_stop" in joined3


@pytest.mark.asyncio
async def test_stream_messages_upstream_error_raises_execution_failure(provider_config):
    """Verify that upstream failures correctly raise ExecutionFailure and do not fail with UnboundLocalError."""
    auth = MagicMock(spec=AntigravityAuthManager)
    auth.is_connected.return_value = True
    auth.access = AsyncMock(
        return_value=AntigravityAccess("ya29.test_token", "test_proj")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b"Rate limit exceeded", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AntigravityProvider(
        provider_config,
        auth=auth,
        admission=immediate_admission(),
        client=client,
    )
    try:
        req = make_messages_request("antigravity/gemini-3.8-flash")
        with pytest.raises(ExecutionFailure) as exc_info:
            async for _ in provider.stream_messages(req):
                pass
        assert exc_info.value.status_code == 429
    finally:
        await provider.cleanup()
        await client.aclose()


