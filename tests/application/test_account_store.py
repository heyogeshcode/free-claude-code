"""Unit tests for AccountStore."""

from pathlib import Path

from free_claude_code.application.account_store import AccountStore


def test_account_lifecycle_and_cooldown(tmp_path: Path) -> None:
    config_file = tmp_path / "accounts.json"
    lock_file = tmp_path / "accounts.lock"
    store = AccountStore(config_path=config_file, lock_path=lock_file)

    # Initially empty or default providers
    assert "antigravity" in store.get_providers()
    assert store.get_accounts("antigravity") == []

    # Add accounts
    acc1 = store.add_account(
        "antigravity",
        label="Account One",
        credentials={"token": "t1"},
        account_id="ag_01",
    )
    acc2 = store.add_account(
        "antigravity",
        label="Account Two",
        credentials={"token": "t2"},
        account_id="ag_02",
    )
    assert acc2.id == "ag_02"

    accounts = store.get_accounts("antigravity")
    assert len(accounts) == 2
    assert accounts[0].id == "ag_01"
    assert accounts[0].priority == 1
    assert accounts[1].id == "ag_02"
    assert accounts[1].priority == 2

    # Model cooldown
    assert not acc1.is_cooling("gemini-3.8-flash")
    store.set_model_cooldown(
        "antigravity", "ag_01", "gemini-3.8-flash", duration_seconds=60
    )
    assert acc1.is_cooling("gemini-3.8-flash")

    # Healthy accounts for model: ag_01 is cooling, so only ag_02
    healthy = store.get_healthy_accounts_for_model("antigravity", "gemini-3.8-flash")
    assert len(healthy) == 1
    assert healthy[0].id == "ag_02"

    # For another model, ag_01 is still healthy
    healthy_other = store.get_healthy_accounts_for_model(
        "antigravity", "claude-3-7-sonnet"
    )
    assert len(healthy_other) == 2


def test_reordering(tmp_path: Path) -> None:
    config_file = tmp_path / "accounts.json"
    lock_file = tmp_path / "accounts.lock"
    store = AccountStore(config_path=config_file, lock_path=lock_file)

    store.add_account("chatgpt", "Acc 1", {}, account_id="c1")
    store.add_account("chatgpt", "Acc 2", {}, account_id="c2")
    store.add_account("chatgpt", "Acc 3", {}, account_id="c3")

    # Move c2 up
    assert store.move_up("chatgpt", "c2") is True
    accounts = store.get_accounts("chatgpt")
    assert [a.id for a in accounts] == ["c2", "c1", "c3"]
    assert [a.priority for a in accounts] == [1, 2, 3]

    # Move c2 down
    assert store.move_down("chatgpt", "c2") is True
    accounts = store.get_accounts("chatgpt")
    assert [a.id for a in accounts] == ["c1", "c2", "c3"]

    # Reload from disk and verify persistence
    store2 = AccountStore(config_path=config_file, lock_path=lock_file)
    reloaded = store2.get_accounts("chatgpt")
    assert [a.id for a in reloaded] == ["c1", "c2", "c3"]
