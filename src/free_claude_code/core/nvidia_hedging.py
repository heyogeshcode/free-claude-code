"""Adaptive Speculative Hedging Engine for NVIDIA NIM & Fallback Multi-Key Pool."""

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.nvidia_key_store import NvidiaKeyStore, get_nvidia_key_store
from free_claude_code.core.nvidia_normalizer import NvidiaSchemaNormalizer

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@dataclass
class HedgingConfig:
    b_max: int = 100
    headroom: float = 1.5
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    max_retries: int = 3


class ActiveConcurrencyTracker:
    """Track concurrent in-flight requests for dynamic load shedding."""

    def __init__(self) -> None:
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return max(1, self._count)

    async def acquire(self) -> None:
        async with self._lock:
            self._count += 1

    async def release(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)


_GLOBAL_CONCURRENCY = ActiveConcurrencyTracker()


def calculate_batch_size(
    healthy_keys_count: int,
    active_concurrency: int,
    b_max: int = 100,
    headroom: float = 1.5,
) -> int:
    """Calculate dynamic speculative racing batch size B."""
    if healthy_keys_count <= 0:
        return 1
    denominator = max(1.0, float(active_concurrency) * float(headroom))
    raw_b = math.floor(healthy_keys_count / denominator)
    return max(1, min(b_max, raw_b))


class NvidiaHedgingEngine:
    """Executes speculative first-wins parallel racing across healthy NVIDIA API keys."""

    def __init__(
        self,
        key_store: NvidiaKeyStore | None = None,
        config: HedgingConfig | None = None,
        concurrency_tracker: ActiveConcurrencyTracker | None = None,
    ) -> None:
        self._key_store = key_store or get_nvidia_key_store()
        self._config = config or HedgingConfig()
        self._concurrency = concurrency_tracker or _GLOBAL_CONCURRENCY

    async def stream_chat_completions(
        self,
        request_body: dict[str, Any],
        *,
        model: str,
        message_id: str,
        is_anthropic_wire: bool = True,
    ) -> AsyncIterator[str]:
        """Race candidate keys concurrently, declare winner on first 200 chunk, stream response."""
        await self._concurrency.acquire()
        try:
            async for chunk in self._execute_hedged_racing(
                request_body,
                model=model,
                message_id=message_id,
                is_anthropic_wire=is_anthropic_wire,
            ):
                yield chunk
        finally:
            await self._concurrency.release()

    async def _execute_hedged_racing(
        self,
        request_body: dict[str, Any],
        *,
        model: str,
        message_id: str,
        is_anthropic_wire: bool,
    ) -> AsyncIterator[str]:
        healthy_keys = self._key_store.get_healthy_keys()
        if not healthy_keys:
            # Refresh from disk in case new keys were added
            self._key_store.discover_and_load()
            healthy_keys = self._key_store.get_healthy_keys()

        if not healthy_keys:
            raise ExecutionFailure(
                kind=FailureKind.AUTHENTICATION,
                status_code=401,
                message="No healthy NVIDIA API keys available in pool",
                retryable=False,
            )

        active_concurrency = self._concurrency.count
        batch_size = calculate_batch_size(
            len(healthy_keys),
            active_concurrency,
            b_max=self._config.b_max,
            headroom=self._config.headroom,
        )

        logger.info(
            "Nvidia hedging: racing batch_size={} across healthy_keys={} active_concurrency={}",
            batch_size,
            len(healthy_keys),
            active_concurrency,
        )

        # Iterate over keys in batches until a winner emerges
        remaining_keys = list(healthy_keys)
        while remaining_keys:
            current_batch = remaining_keys[:batch_size]
            remaining_keys = remaining_keys[batch_size:]

            winner = await self._race_batch(current_batch, request_body)
            if winner is not None:
                if len(winner) == 6:
                    client, response, winning_key, first_chunk, line_iter, start_time = winner
                else:
                    client, response, winning_key, first_chunk, start_time = winner
                    line_iter = response.aiter_lines()
                # Record success and latency
                latency_ms = (time.time() - start_time) * 1000.0
                self._key_store.record_success(winning_key, latency_ms)
                logger.info(
                    "Nvidia hedging winner: key=...{} latency={:.1f}ms",
                    winning_key[-6:],
                    latency_ms,
                )

                try:
                    state: dict[str, Any] = {}
                    # Yield initial chunk
                    if is_anthropic_wire:
                        for sse in self._convert_raw_chunk(
                            first_chunk, state, model, message_id
                        ):
                            yield sse
                    else:
                        yield first_chunk

                    # Stream remainder of winner response
                    async for line in line_iter:
                        if not line:
                            continue
                        if is_anthropic_wire:
                            for sse in self._convert_raw_chunk(
                                line, state, model, message_id
                            ):
                                yield sse
                        else:
                            yield f"{line}\n\n"

                    # Close any open blocks if not closed
                    if is_anthropic_wire and not state.get("block_closed", False):
                        if state.get("block_type") is not None:
                            stop_ev = {
                                "type": "content_block_stop",
                                "index": state.get("current_block_index", 0),
                            }
                            yield f"event: content_block_stop\ndata: {json.dumps(stop_ev)}\n\n"
                        msg_delta = {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 1},
                        }
                        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        state["block_closed"] = True

                    return
                finally:
                    await response.aclose()
                    await client.aclose()

        # If all batches exhausted
        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="All candidate NVIDIA NIM keys exhausted or rate-limited",
            retryable=True,
        )

    async def _race_batch(
        self,
        keys: list[str],
        request_body: dict[str, Any],
    ) -> tuple[httpx.AsyncClient, httpx.Response, str, str, Any, float] | None:
        """Race keys concurrently. First key to return HTTP 200 and a chunk wins.

        All losing connections/sockets are immediately aborted.
        """
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        tasks: list[asyncio.Task] = []
        clients_map: dict[str, tuple[httpx.AsyncClient, httpx.Response | None]] = {}

        async def runner(key: str) -> None:
            t0 = time.time()
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._config.connect_timeout_seconds,
                    read=self._config.read_timeout_seconds,
                    write=10.0,
                    pool=10.0,
                )
            )
            clients_map[key] = (client, None)
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            try:
                req = client.build_request(
                    "POST", NVIDIA_API_URL, headers=headers, json=request_body
                )
                response = await client.send(req, stream=True)
                clients_map[key] = (client, response)

                if response.status_code == 200:
                    line_iter = response.aiter_lines()
                    first_line = None
                    # Read first line to confirm live stream
                    async for line in line_iter:
                        if line and line.startswith("data:"):
                            first_line = line
                            break
                    if first_line is not None:
                        await queue.put(
                            ("WIN", (client, response, key, first_line, line_iter, t0))
                        )
                        return
                    # If stream ended with no chunk
                    await queue.put(("FAIL", (key, 502, "Empty stream")))
                elif response.status_code == 429:
                    self._key_store.record_rate_limit(key)
                    await queue.put(("FAIL", (key, 429, "Rate limited")))
                elif response.status_code in (401, 403):
                    self._key_store.record_failure(key, unrecoverable=True)
                    await queue.put(("FAIL", (key, response.status_code, "Auth error")))
                else:
                    self._key_store.record_failure(key)
                    await queue.put(
                        (
                            "FAIL",
                            (
                                key,
                                response.status_code,
                                f"Status {response.status_code}",
                            ),
                        )
                    )
            except TimeoutError, httpx.TimeoutException:
                self._key_store.record_hang(key)
                await queue.put(("FAIL", (key, 504, "Timeout")))
            except Exception as e:
                self._key_store.record_failure(key)
                await queue.put(("FAIL", (key, 500, str(e))))

        for k in keys:
            t = asyncio.create_task(runner(k))
            tasks.append(t)

        pending = len(tasks)
        winning_result = None

        while pending > 0:
            status, payload = await queue.get()
            if status == "WIN":
                winning_result = payload
                winning_key = payload[2]
                # CANCEL ALL OTHER TASKS IMMEDIATELY
                for t in tasks:
                    if not t.done():
                        t.cancel()
                # ABORT AND CLOSE ALL LOSING CLIENTS/RESPONSES
                for k, (cli, resp) in clients_map.items():
                    if k != winning_key:
                        try:
                            if resp is not None:
                                asyncio.create_task(resp.aclose())
                            asyncio.create_task(cli.aclose())
                        except Exception:
                            pass
                break
            else:
                pending -= 1
                key = payload[0]
                cli, resp = clients_map.get(key, (None, None))
                if resp is not None:
                    await resp.aclose()
                if cli is not None:
                    await cli.aclose()

        # Clean up any cancelled tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        return winning_result

    def _convert_raw_chunk(
        self,
        raw_line: str,
        state: dict[str, Any],
        model: str,
        message_id: str,
    ) -> list[str]:
        """Parse raw SSE data line and convert to Anthropic SSE event strings."""
        line = raw_line.strip()
        if not line:
            return []
        if line.startswith("data: "):
            payload = line[6:].strip()
        elif line.startswith("data:"):
            payload = line[5:].strip()
        else:
            return []

        if payload == "[DONE]":
            events = []
            if state.get("block_type") is not None:
                stop_ev = {
                    "type": "content_block_stop",
                    "index": state.get("current_block_index", 0),
                }
                events.append(
                    f"event: content_block_stop\ndata: {json.dumps(stop_ev)}\n\n"
                )
                state["block_type"] = None
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            }
            events.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n")
            events.append(
                f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            )
            state["block_closed"] = True
            return events

        try:
            chunk_json = json.loads(payload)
            return NvidiaSchemaNormalizer.openai_chunk_to_anthropic_events(
                chunk_json, state, model=model, message_id=message_id
            )
        except json.JSONDecodeError:
            return []


_GLOBAL_HEDGING_ENGINE: NvidiaHedgingEngine | None = None


def get_nvidia_hedging_engine() -> NvidiaHedgingEngine:
    global _GLOBAL_HEDGING_ENGINE
    if _GLOBAL_HEDGING_ENGINE is None:
        _GLOBAL_HEDGING_ENGINE = NvidiaHedgingEngine()
    return _GLOBAL_HEDGING_ENGINE
