"""Unit tests for /v1/nvidia_nim and /v1/nvidia_fallback dual routes."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


@pytest.mark.asyncio
async def test_nvidia_fallback_route_anthropic_format():
    app = create_test_app(Settings())

    async def mock_stream(*args, **kwargs):
        yield "event: message_start\ndata: {}\n\n"
        yield "event: content_block_delta\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    mock_engine = AsyncMock()
    mock_engine.stream_chat_completions = mock_stream

    with patch(
        "free_claude_code.api.routes.get_nvidia_hedging_engine",
        return_value=mock_engine,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/nvidia_fallback",
                headers={"anthropic-version": "2023-06-01"},
                json={
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 100,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            text = resp.text
            assert "event: message_start" in text
            assert "event: message_stop" in text


@pytest.mark.asyncio
async def test_nvidia_fallback_route_openai_format():
    app = create_test_app(Settings())

    async def mock_stream(*args, **kwargs):
        yield 'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        yield "data: [DONE]\n\n"

    mock_engine = AsyncMock()
    mock_engine.stream_chat_completions = mock_stream

    with patch(
        "free_claude_code.api.routes.get_nvidia_hedging_engine",
        return_value=mock_engine,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/nvidia_fallback/chat/completions",
                json={
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            text = resp.text
            assert "Hi" in text
