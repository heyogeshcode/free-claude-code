"""Tests for Antigravity dual-pool quota tracking and serialization."""

import pytest
from free_claude_code.providers.antigravity.quota import (
    ANTHROPIC_GPT_POOL_ID,
    GEMINI_POOL_ID,
    AntigravityPoolQuota,
    parse_quota_response,
)
from free_claude_code.api.admin_routes import _serialize_accounts
from free_claude_code.application.account_store import Account, AccountStore
import free_claude_code.application.account_store as account_store_module


def test_parse_quota_response_dual_pools():
    payload = {
        "buckets": [
            {
                "remainingFraction": 0.95,
                "resetTime": "2026-09-21T16:01:00Z",
                "modelIds": [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3.8-flash-tiered",
                ],
            },
            {
                "remainingFraction": 0.28,
                "resetTime": "2026-09-21T14:55:52Z",
                "modelIds": [
                    "claude-opus-4-6-thinking",
                    "claude-sonnet-4-6",
                    "gpt-oss-120b-medium",
                ],
            },
        ]
    }

    pools = parse_quota_response(payload)
    assert GEMINI_POOL_ID in pools
    assert ANTHROPIC_GPT_POOL_ID in pools

    gemini = pools[GEMINI_POOL_ID]
    assert gemini.remaining_pct == 95
    assert gemini.remaining_fraction == 0.95
    assert gemini.reset_time == "2026-09-21T16:01:00Z"
    assert "gemini-3.8-flash-tiered" in gemini.models

    anthropic = pools[ANTHROPIC_GPT_POOL_ID]
    assert anthropic.remaining_pct == 28
    assert anthropic.remaining_fraction == 0.28
    assert anthropic.reset_time == "2026-09-21T14:55:52Z"
    assert "claude-sonnet-4-6" in anthropic.models
    assert "claude-opus-4-6-thinking" in anthropic.models


def test_parse_quota_response_empty_defaults():
    pools = parse_quota_response({})
    assert GEMINI_POOL_ID in pools
    assert ANTHROPIC_GPT_POOL_ID in pools
    assert pools[GEMINI_POOL_ID].remaining_pct == 100
    assert pools[ANTHROPIC_GPT_POOL_ID].remaining_pct == 100


def test_serialize_accounts_with_dual_pool_quotas(tmp_path, monkeypatch):
    config_file = tmp_path / "accounts.json"
    lock_file = tmp_path / "accounts.lock"
    store = AccountStore(config_path=config_file, lock_path=lock_file)
    monkeypatch.setattr(account_store_module, "_GLOBAL_ACCOUNT_STORE", store)

    store.add_account("antigravity", "user1@gmail.com", {}, account_id="acc1")
    store.add_account("antigravity", "user2@gmail.com", {}, account_id="acc2")

    mock_quotas = {
        "acc1": {
            GEMINI_POOL_ID: AntigravityPoolQuota(
                pool_id=GEMINI_POOL_ID,
                name="Gemini Models",
                remaining_fraction=0.90,
                remaining_pct=90,
                reset_time="2026-09-21T16:00:00Z",
                models=["gemini-2.5-flash"],
            ),
            ANTHROPIC_GPT_POOL_ID: AntigravityPoolQuota(
                pool_id=ANTHROPIC_GPT_POOL_ID,
                name="Anthropic + GPT Models",
                remaining_fraction=0.30,
                remaining_pct=30,
                reset_time="2026-09-21T14:00:00Z",
                models=["claude-sonnet-4-6"],
            ),
        },
        "acc2": {
            GEMINI_POOL_ID: AntigravityPoolQuota(
                pool_id=GEMINI_POOL_ID,
                name="Gemini Models",
                remaining_fraction=0.80,
                remaining_pct=80,
                reset_time="2026-09-21T16:00:00Z",
                models=["gemini-2.5-flash"],
            ),
            ANTHROPIC_GPT_POOL_ID: AntigravityPoolQuota(
                pool_id=ANTHROPIC_GPT_POOL_ID,
                name="Anthropic + GPT Models",
                remaining_fraction=0.50,
                remaining_pct=50,
                reset_time="2026-09-21T14:00:00Z",
                models=["claude-sonnet-4-6"],
            ),
        },
    }

    serialized = _serialize_accounts("antigravity", quotas_by_account=mock_quotas)
    assert serialized["provider_id"] == "antigravity"
    assert len(serialized["accounts"]) == 2

    # Check acc1
    acc1 = next(a for a in serialized["accounts"] if a["id"] == "acc1")
    assert acc1["remaining_pct"] == 30.0  # reflects Anthropic quota
    assert "pools" in acc1
    assert len(acc1["pools"]) == 2

    # Check limit_summary
    summary = serialized["limit_summary"]
    assert "pools" in summary
    pools = summary["pools"]
    assert len(pools) == 2

    gemini_summary = next(p for p in pools if p["pool_id"] == GEMINI_POOL_ID)
    anthropic_summary = next(p for p in pools if p["pool_id"] == ANTHROPIC_GPT_POOL_ID)

    assert gemini_summary["remaining_pct"] == 85  # (90 + 80) / 2
    assert anthropic_summary["remaining_pct"] == 40  # (30 + 50) / 2
