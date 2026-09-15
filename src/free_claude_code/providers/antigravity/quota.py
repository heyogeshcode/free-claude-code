"""Antigravity quota tracking and dual-pool management (Gemini vs Anthropic/GPT)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from .models import DEFAULT_MODELS_HOSTS

GEMINI_POOL_ID = "gemini"
ANTHROPIC_GPT_POOL_ID = "anthropic_gpt"

GEMINI_POOL_NAME = "Gemini Models"
ANTHROPIC_GPT_POOL_NAME = "Anthropic + GPT Models"


@dataclass(frozen=True, slots=True)
class AntigravityPoolQuota:
    pool_id: str
    name: str
    remaining_fraction: float
    remaining_pct: int
    reset_time: str | None
    models: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "name": self.name,
            "remaining_fraction": self.remaining_fraction,
            "remaining_pct": self.remaining_pct,
            "reset_time": self.reset_time,
            "models": list(self.models),
        }


@dataclass
class AccountQuotaState:
    account_id: str
    fetched_at: float
    pools: dict[str, AntigravityPoolQuota]


_QUOTA_CACHE: dict[str, AccountQuotaState] = {}
_CACHE_TTL_SECONDS = 45.0


def parse_quota_response(data: dict[str, Any]) -> dict[str, AntigravityPoolQuota]:
    """Parse upstream /v1internal:retrieveUserQuota JSON into dual pools."""
    buckets = data.get("buckets", [])
    if not isinstance(buckets, list):
        return _default_pools()

    pools: dict[str, AntigravityPoolQuota] = {}

    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue

        raw_fraction = bucket.get("remainingFraction")
        fraction = float(raw_fraction) if isinstance(raw_fraction, (int, float)) else 1.0
        pct = max(0, min(100, round(fraction * 100)))
        reset_time = bucket.get("resetTime")
        if not isinstance(reset_time, str):
            reset_time = None

        raw_models = bucket.get("modelIds", [])
        models = [str(m) for m in raw_models if isinstance(m, str)]

        # Determine pool category based on models contained
        is_anthropic_gpt = any(
            m.startswith("claude-") or m.startswith("gpt-") or "opus" in m or "sonnet" in m
            for m in models
        )

        pool_id = ANTHROPIC_GPT_POOL_ID if is_anthropic_gpt else GEMINI_POOL_ID
        name = ANTHROPIC_GPT_POOL_NAME if is_anthropic_gpt else GEMINI_POOL_NAME

        pools[pool_id] = AntigravityPoolQuota(
            pool_id=pool_id,
            name=name,
            remaining_fraction=fraction,
            remaining_pct=pct,
            reset_time=reset_time,
            models=models,
        )

    # Ensure both pools exist with defaults if missing
    if GEMINI_POOL_ID not in pools:
        pools[GEMINI_POOL_ID] = AntigravityPoolQuota(
            pool_id=GEMINI_POOL_ID,
            name=GEMINI_POOL_NAME,
            remaining_fraction=1.0,
            remaining_pct=100,
            reset_time=None,
            models=[],
        )
    if ANTHROPIC_GPT_POOL_ID not in pools:
        pools[ANTHROPIC_GPT_POOL_ID] = AntigravityPoolQuota(
            pool_id=ANTHROPIC_GPT_POOL_ID,
            name=ANTHROPIC_GPT_POOL_NAME,
            remaining_fraction=1.0,
            remaining_pct=100,
            reset_time=None,
            models=[],
        )

    return pools


def _default_pools() -> dict[str, AntigravityPoolQuota]:
    return {
        GEMINI_POOL_ID: AntigravityPoolQuota(
            pool_id=GEMINI_POOL_ID,
            name=GEMINI_POOL_NAME,
            remaining_fraction=1.0,
            remaining_pct=100,
            reset_time=None,
            models=[],
        ),
        ANTHROPIC_GPT_POOL_ID: AntigravityPoolQuota(
            pool_id=ANTHROPIC_GPT_POOL_ID,
            name=ANTHROPIC_GPT_POOL_NAME,
            remaining_fraction=1.0,
            remaining_pct=100,
            reset_time=None,
            models=[],
        ),
    }


def get_cached_account_quota(account_id: str) -> dict[str, AntigravityPoolQuota] | None:
    """Return cached quota if still valid under TTL."""
    entry = _QUOTA_CACHE.get(account_id)
    if entry and (time.time() - entry.fetched_at) < _CACHE_TTL_SECONDS:
        return entry.pools
    return None


async def fetch_account_quota(
    client: httpx.AsyncClient,
    access_token: str,
    account_id: str,
    project_id: str | None = None,
    force_refresh: bool = False,
) -> dict[str, AntigravityPoolQuota]:
    """Fetch live quota buckets from Google Antigravity backend."""
    if not force_refresh:
        cached = get_cached_account_quota(account_id)
        if cached is not None:
            return cached

    hosts = ("cloudcode-pa.googleapis.com", *DEFAULT_MODELS_HOSTS)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "Antigravity",
    }
    body: dict[str, Any] = {}
    if project_id:
        body["project"] = project_id

    for host in hosts:
        url = f"https://{host}/v1internal:retrieveUserQuota"
        try:
            res = await client.post(url, json=body, headers=headers, timeout=5.0)
            if res.is_success:
                data = res.json()
                pools = parse_quota_response(data)
                _QUOTA_CACHE[account_id] = AccountQuotaState(
                    account_id=account_id,
                    fetched_at=time.time(),
                    pools=pools,
                )
                return pools
            else:
                logger.debug(
                    "retrieveUserQuota returned status {} from {}",
                    res.status_code,
                    host,
                )
        except Exception as exc:
            logger.debug("Failed retrieveUserQuota from {}: {}", host, exc)

    # If all hosts fail, return stale cache or default pools
    if account_id in _QUOTA_CACHE:
        return _QUOTA_CACHE[account_id].pools

    fallback = _default_pools()
    _QUOTA_CACHE[account_id] = AccountQuotaState(
        account_id=account_id,
        fetched_at=time.time(),
        pools=fallback,
    )
    return fallback
