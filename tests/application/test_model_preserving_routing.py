"""Test model-preserving multi-account routing and failover."""

from free_claude_code.application.account_store import AccountStore
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings


def test_model_preserving_multi_account_ordering(tmp_path):
    config_file = tmp_path / "accounts.json"
    store = AccountStore(config_file)

    # Register 3 Antigravity accounts with explicit priorities
    store.add_account(
        provider_id="antigravity",
        label="Account 1",
        account_id="acc_01",
        credentials={"token": "t1"},
        priority=1,
    )
    store.add_account(
        provider_id="antigravity",
        label="Account 2",
        account_id="acc_02",
        credentials={"token": "t2"},
        priority=2,
    )
    store.add_account(
        provider_id="antigravity",
        label="Account 3",
        account_id="acc_03",
        credentials={"token": "t3"},
        priority=3,
    )

    settings = Settings(
        MODEL="antigravity/gemini-3.8-flash",
        MODEL_FALLBACKS="antigravity/gemini-3.0-flash,groq/default",
    )

    router = ModelRouter(settings, account_store=store)
    route = router.resolve("claude-sonnet-4")

    # Primary should be Account 1 on gemini-3.8-flash
    assert route.primary.provider_id == "antigravity"
    assert route.primary.provider_model == "gemini-3.8-flash"
    assert route.primary.account_id == "acc_01"

    # Fallbacks should first be Account 2 and Account 3 on the SAME model (model-preserving)
    # followed by accounts for degradation fallback targets
    same_model_fallbacks = [
        f for f in route.fallbacks if f.provider_model == "gemini-3.8-flash"
    ]
    assert len(same_model_fallbacks) == 2
    assert same_model_fallbacks[0].account_id == "acc_02"
    assert same_model_fallbacks[1].account_id == "acc_03"

    # Degraded models come AFTER same-model accounts are exhausted
    degraded_models = [
        f for f in route.fallbacks if f.provider_model != "gemini-3.8-flash"
    ]
    assert len(degraded_models) >= 1
    assert degraded_models[0].provider_model == "gemini-3.0-flash"


def test_cooldown_excludes_rate_limited_account(tmp_path):
    config_file = tmp_path / "accounts.json"
    store = AccountStore(config_file)

    store.add_account(
        provider_id="antigravity",
        label="Primary",
        account_id="acc_primary",
        credentials={"token": "p"},
        priority=1,
    )
    store.add_account(
        provider_id="antigravity",
        label="Secondary",
        account_id="acc_secondary",
        credentials={"token": "s"},
        priority=2,
    )

    # Set 60s cooldown for acc_primary on gemini-3.8-flash
    store.set_model_cooldown("antigravity", "acc_primary", "gemini-3.8-flash", 60.0)

    settings = Settings(
        MODEL="antigravity/gemini-3.8-flash",
        MODEL_FALLBACKS="antigravity/gemini-3.0-flash",
    )
    router = ModelRouter(settings, account_store=store)
    route = router.resolve("antigravity/gemini-3.8-flash")

    # Primary should have rotated directly to acc_secondary since acc_primary is cooling
    assert route.primary.account_id == "acc_secondary"
