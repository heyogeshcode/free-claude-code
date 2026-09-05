import base64
import json
from pathlib import Path

import httpx
import pytest

from free_claude_code.application.connected_accounts import (
    ConnectedAccountState,
)
from free_claude_code.providers.antigravity.auth import (
    AntigravityAuthManager,
    AntigravityReconnectRequired,
)


def _fake_jwt(payload: dict[str, object]) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    return f"header.{encoded}.sig"


def _credential_payload(
    *, expires_at: int, email: str = "test@example.com"
) -> dict[str, object]:
    id_token = _fake_jwt({"email": email, "sub": "12345"})
    return {
        "version": 1,
        "credentials": {
            "access_token": "ya29.test-access-token",
            "refresh_token": "1//test-refresh-token",
            "id_token": id_token,
            "email": email,
            "expires_at": expires_at,
            "project_id": "aicode-consumers",
        },
    }


@pytest.mark.asyncio
async def test_auth_manager_initial_state_disconnected(tmp_path: Path) -> None:
    manager = AntigravityAuthManager(
        credential_path=tmp_path / "auth" / "antigravity.json",
        lock_path=tmp_path / "auth" / "antigravity.lock",
        cli_token_path=tmp_path / "nonexistent",
    )
    try:
        assert not manager.is_connected()
        status = manager.status()
        assert status.state == ConnectedAccountState.DISCONNECTED
        assert not status.connected
        assert status.email is None
        assert status.provider_id == "antigravity"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_auth_manager_loads_existing_credentials(tmp_path: Path) -> None:
    cred_file = tmp_path / "auth" / "antigravity.json"
    cred_file.parent.mkdir(parents=True)
    cred_file.write_text(
        json.dumps(_credential_payload(expires_at=9999999999, email="user@gmail.com"))
    )

    manager = AntigravityAuthManager(
        credential_path=cred_file,
        lock_path=tmp_path / "auth" / "antigravity.lock",
        cli_token_path=tmp_path / "nonexistent",
    )
    try:
        assert manager.is_connected()
        status = manager.status()
        assert status.state == ConnectedAccountState.CONNECTED
        assert status.connected
        assert status.email == "user@gmail.com"

        # Status must never expose secrets
        safe_dict = status.as_dict()
        assert "access_token" not in safe_dict
        assert "refresh_token" not in safe_dict
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_auth_manager_auto_refreshes_expired_token(tmp_path: Path) -> None:
    cred_file = tmp_path / "auth" / "antigravity.json"
    cred_file.parent.mkdir(parents=True)
    # Expired token
    cred_file.write_text(
        json.dumps(_credential_payload(expires_at=1, email="user@gmail.com"))
    )

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "ya29.new-refreshed-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = AntigravityAuthManager(
        credential_path=cred_file,
        lock_path=tmp_path / "auth" / "antigravity.lock",
        cli_token_path=tmp_path / "nonexistent",
        client=client,
    )
    try:
        access = await manager.access()
        assert access.access_token == "ya29.new-refreshed-token"
        assert access.project_id == "aicode-consumers"
        assert len(requests) == 1
        assert "refresh_token" in requests[0].read().decode()

        # Verify updated file
        saved = json.loads(cred_file.read_text())
        assert saved["credentials"]["access_token"] == "ya29.new-refreshed-token"
    finally:
        await manager.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_auth_manager_disconnect(tmp_path: Path) -> None:
    cred_file = tmp_path / "auth" / "antigravity.json"
    cred_file.parent.mkdir(parents=True)
    cred_file.write_text(json.dumps(_credential_payload(expires_at=9999999999)))

    manager = AntigravityAuthManager(
        credential_path=cred_file,
        lock_path=tmp_path / "auth" / "antigravity.lock",
        cli_token_path=tmp_path / "nonexistent",
    )
    try:
        assert manager.is_connected()

        status = await manager.disconnect()
        assert not manager.is_connected()
        assert status.state == ConnectedAccountState.DISCONNECTED
        assert not cred_file.is_file()

        with pytest.raises(AntigravityReconnectRequired):
            await manager.access()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_auth_manager_seeds_from_native_cli_token(tmp_path: Path) -> None:
    cli_token_file = tmp_path / "antigravity-oauth-token"
    cli_token_file.write_text(
        json.dumps(
            {
                "auth_method": "consumer",
                "token": {
                    "access_token": "ya29.from-cli-token",
                    "refresh_token": "1//from-cli-token",
                    "id_token": _fake_jwt({"email": "seeded@gmail.com"}),
                    "expiry": "2099-01-01T00:00:00Z",
                },
            }
        )
    )

    dest_file = tmp_path / "auth" / "antigravity.json"
    manager = AntigravityAuthManager(
        credential_path=dest_file,
        lock_path=tmp_path / "auth" / "antigravity.lock",
        cli_token_path=cli_token_file,
    )
    try:
        assert manager.is_connected()
        assert manager.status().email == "seeded@gmail.com"
        access = await manager.access()
        assert access.access_token == "ya29.from-cli-token"
        assert dest_file.is_file()
    finally:
        await manager.close()
