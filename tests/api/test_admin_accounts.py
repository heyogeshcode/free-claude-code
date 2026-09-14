"""Tests for multi-account management REST endpoints in /admin."""

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.account_store import AccountStore
import free_claude_code.application.account_store as account_store_module
from tests.api.support import create_test_app


def _local_client(app):
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_admin_multi_account_lifecycle(tmp_path, monkeypatch):
    config_file = tmp_path / "accounts.json"
    lock_file = tmp_path / "accounts.lock"
    store = AccountStore(config_path=config_file, lock_path=lock_file)
    monkeypatch.setattr(account_store_module, "_GLOBAL_ACCOUNT_STORE", store)

    app = create_test_app()
    client = _local_client(app)

    # 1. Initially empty
    resp = client.get("/admin/api/providers/antigravity/accounts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider_id"] == "antigravity"
    assert data["accounts"] == []

    # 2. Add accounts to store
    store.add_account("antigravity", "Work Account", {"tok": "1"}, account_id="ag_work")
    store.add_account("antigravity", "Personal Account", {"tok": "2"}, account_id="ag_personal")
    store.add_account("antigravity", "Backup Account", {"tok": "3"}, account_id="ag_backup")

    resp = client.get("/admin/api/providers/antigravity/accounts")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["accounts"]) == 3
    assert [a["id"] for a in data["accounts"]] == ["ag_work", "ag_personal", "ag_backup"]
    assert [a["priority"] for a in data["accounts"]] == [1, 2, 3]

    # 3. Move ag_personal up (swap with ag_work)
    resp = client.post("/admin/api/providers/antigravity/accounts/ag_personal/move-up")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert [a["id"] for a in data["accounts"]] == ["ag_personal", "ag_work", "ag_backup"]
    assert [a["priority"] for a in data["accounts"]] == [1, 2, 3]

    # 4. Move ag_personal down (swap with ag_work back)
    resp = client.post("/admin/api/providers/antigravity/accounts/ag_personal/move-down")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert [a["id"] for a in data["accounts"]] == ["ag_work", "ag_personal", "ag_backup"]

    # 5. Set and clear cooldown
    store.set_model_cooldown("antigravity", "ag_work", "gemini-3.8-flash", duration_seconds=60)
    resp = client.get("/admin/api/providers/antigravity/accounts")
    data = resp.json()
    work_acc = next(a for a in data["accounts"] if a["id"] == "ag_work")
    assert work_acc["is_cooling"] is True
    assert len(work_acc["cooling_models"]) == 1
    assert work_acc["cooling_models"][0]["model"] == "gemini-3.8-flash"

    # Clear cooldown
    resp = client.post("/admin/api/providers/antigravity/accounts/ag_work/clear-cooldown")
    assert resp.status_code == 200
    data = resp.json()
    work_acc = next(a for a in data["accounts"] if a["id"] == "ag_work")
    assert work_acc["is_cooling"] is False
    assert len(work_acc["cooling_models"]) == 0

    # 6. Delete account
    resp = client.delete("/admin/api/providers/antigravity/accounts/ag_personal")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["accounts"]) == 2
    assert [a["id"] for a in data["accounts"]] == ["ag_work", "ag_backup"]
    assert [a["priority"] for a in data["accounts"]] == [1, 2]


def test_admin_account_email_resolution():
    import base64
    import json
    from free_claude_code.api.admin_routes import _resolve_account_email
    from free_claude_code.application.account_store import Account

    # 1. Label is already an email
    acc1 = Account(id="1", label="dev@example.com", priority=1, credentials={})
    assert _resolve_account_email(acc1) == "dev@example.com"

    # 2. Email inside credentials dict
    acc2 = Account(id="2", label="ChatGPT Plus", priority=1, credentials={"email": "plus_user@openai.com"})
    assert _resolve_account_email(acc2) == "plus_user@openai.com"

    # 3. Email inside credentials.credentials dict
    acc3 = Account(id="3", label="Antigravity", priority=1, credentials={"credentials": {"email": "google_user@gmail.com"}})
    assert _resolve_account_email(acc3) == "google_user@gmail.com"

    # 4. Email inside JWT id_token
    claims = {"email": "jwt_user@company.org", "sub": "12345"}
    encoded_claims = base64.urlsafe_b64encode(json.dumps(claims).encode("utf-8")).decode("utf-8").rstrip("=")
    fake_jwt = f"header.{encoded_claims}.signature"
    acc4 = Account(id="4", label="My JWT Account", priority=1, credentials={"id_token": fake_jwt})
    assert _resolve_account_email(acc4) == "jwt_user@company.org"

    # 5. Email inside OpenAI custom profile claim
    openai_claims = {"https://api.openai.com/profile": {"email": "chatgpt_user@gmail.com"}}
    encoded_oa = base64.urlsafe_b64encode(json.dumps(openai_claims).encode("utf-8")).decode("utf-8").rstrip("=")
    fake_oa_jwt = f"header.{encoded_oa}.signature"
    acc5 = Account(id="5", label="OpenAI Account", priority=1, credentials={"id_token": fake_oa_jwt})
    assert _resolve_account_email(acc5) == "chatgpt_user@gmail.com"


def test_admin_oauth_limits_endpoint(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from free_claude_code.application.connected_accounts import (
        ConnectedAccountState,
        ConnectedAccountStatus,
    )

    config_file = tmp_path / "accounts.json"
    lock_file = tmp_path / "accounts.lock"
    store = AccountStore(config_path=config_file, lock_path=lock_file)
    monkeypatch.setattr(account_store_module, "_GLOBAL_ACCOUNT_STORE", store)

    store.add_account("antigravity", "ag1@gmail.com", {}, account_id="ag1")
    store.add_account("antigravity", "ag2@gmail.com", {}, account_id="ag2")
    store.add_account("openai", "o1@openai.com", {}, account_id="o1")

    mock_ag = MagicMock()
    mock_ag.is_connected.return_value = True
    mock_ag.status.return_value = ConnectedAccountStatus(
        provider_id="antigravity",
        state=ConnectedAccountState.CONNECTED,
        connected=True,
        revision=1,
    )

    app = create_test_app(connected_accounts={"antigravity": mock_ag})
    client = _local_client(app)

    resp = client.get("/admin/api/oauth/limits")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data
    assert len(data["providers"]) == 1
    ag = data["providers"][0]
    assert ag["provider_id"] == "antigravity"
    assert ag["total_accounts"] == 2
    assert ag["active_accounts"] == 2
    assert ag["remaining_pct"] == 100
    assert ag["display_name"] == "Google Antigravity"

