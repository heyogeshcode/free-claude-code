"""Unit tests for NvidiaHedgingEngine and speculative parallel racing."""

from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.nvidia_hedging import (
    ActiveConcurrencyTracker,
    NvidiaHedgingEngine,
    calculate_batch_size,
)
from free_claude_code.core.nvidia_key_store import NvidiaKeyStore


def test_calculate_batch_size_load_shedding():
    # Low traffic, many healthy keys -> capped at b_max
    assert (
        calculate_batch_size(
            healthy_keys_count=1000, active_concurrency=1, b_max=20, headroom=1.5
        )
        == 20
    )
    assert (
        calculate_batch_size(
            healthy_keys_count=1000, active_concurrency=1, b_max=100, headroom=1.5
        )
        == 100
    )

    # High concurrency -> reduced batch size
    # K=150, C=10, H=1.5 -> denominator = 15 -> B = 10
    assert (
        calculate_batch_size(
            healthy_keys_count=150, active_concurrency=10, b_max=50, headroom=1.5
        )
        == 10
    )

    # Very high concurrency / few keys -> floor at 1
    assert (
        calculate_batch_size(
            healthy_keys_count=5, active_concurrency=20, b_max=20, headroom=1.5
        )
        == 1
    )
    assert (
        calculate_batch_size(
            healthy_keys_count=0, active_concurrency=1, b_max=20, headroom=1.5
        )
        == 1
    )


@pytest.mark.asyncio
async def test_concurrency_tracker():
    tracker = ActiveConcurrencyTracker()
    assert tracker.count == 1

    await tracker.acquire()
    assert tracker.count == 1

    await tracker.acquire()
    assert tracker.count == 2

    await tracker.release()
    assert tracker.count == 1

    await tracker.release()
    assert tracker.count == 1


@pytest.mark.asyncio
async def test_speculative_racing_winner_aborts_losers(tmp_path):
    key_file = tmp_path / "nvidia_working.txt"
    state_file = tmp_path / "nvidia_state.json"
    key_file.write_text("nvapi-winner-key-1234567890\nnvapi-loser-key-1234567890\n")

    store = NvidiaKeyStore(key_file, state_file)
    store.discover_and_load()

    engine = NvidiaHedgingEngine(key_store=store)

    # Mock _race_batch to verify winner declaration and metrics update
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200

    async def mock_aiter_lines():
        yield 'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}'
        yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'
        yield "data: [DONE]"

    mock_response.aiter_lines = mock_aiter_lines

    winner_tuple = (
        mock_client,
        mock_response,
        "nvapi-winner-key-1234567890",
        'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
        100.0,
    )

    with patch.object(engine, "_race_batch", return_value=winner_tuple):
        chunks = [
            chunk
            async for chunk in engine.stream_chat_completions(
                {
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                model="meta/llama-3.3-70b-instruct",
                message_id="msg_test",
                is_anthropic_wire=True,
            )
        ]

        assert len(chunks) > 0
        assert any("event: message_start" in c for c in chunks)
        assert any("event: content_block_delta" in c for c in chunks)
        assert any("event: message_stop" in c for c in chunks)

        # Winning key health recorded
        health = store.get_key_health("nvapi-winner-key-1234567890")
        assert health is not None
        assert health.consecutive_successes == 1
        assert health.total_requests == 1


@pytest.mark.asyncio
async def test_speculative_racing_no_stream_consumed(tmp_path):
    """Verify that using real httpx.Response does not raise httpx.StreamConsumed."""
    import httpx

    key_file = tmp_path / "nvidia_working.txt"
    state_file = tmp_path / "nvidia_state.json"
    key_file.write_text("nvapi-real-stream-key-1234567890\n")

    store = NvidiaKeyStore(key_file, state_file)
    store.discover_and_load()

    engine = NvidiaHedgingEngine(key_store=store)

    stream_content = (
        b'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "World!"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    real_response = httpx.Response(200, stream=httpx.ByteStream(stream_content))
    line_iter = real_response.aiter_lines()
    first_chunk = None
    async for l in line_iter:
        if l.startswith("data:"):
            first_chunk = l
            break

    mock_client = AsyncMock()
    winner_tuple = (
        mock_client,
        real_response,
        "nvapi-real-stream-key-1234567890",
        first_chunk,
        line_iter,
        100.0,
    )

    with patch.object(engine, "_race_batch", return_value=winner_tuple):
        chunks = [
            chunk
            async for chunk in engine.stream_chat_completions(
                {
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                model="meta/llama-3.3-70b-instruct",
                message_id="msg_test",
                is_anthropic_wire=True,
            )
        ]

    assert len(chunks) > 0
    text_deltas = [c for c in chunks if "content_block_delta" in c]
    assert len(text_deltas) >= 2

