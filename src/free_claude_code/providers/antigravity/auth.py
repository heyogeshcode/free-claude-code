"""Antigravity credential lifecycle, OAuth state, token refresh, and persistence."""

import asyncio
import base64
import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.config.paths import (
    antigravity_auth_lock_path,
    antigravity_auth_path,
    antigravity_cli_token_path,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

from .login import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    GOOGLE_OAUTH_TOKEN_URL,
    AntigravityLoginError,
    BrowserAuthorization,
    exchange_authorization_code,
)

REFRESH_EARLY_SECONDS = 5 * 60
DEFAULT_PROJECT_ID = "aicode-consumers"


class AntigravityReconnectRequired(RuntimeError):
    """The saved Antigravity session is absent or expired."""


@dataclass(frozen=True, slots=True, repr=False)
class AntigravityAccess:
    """Current upstream authorization credentials."""

    access_token: str
    project_id: str


@dataclass(frozen=True, slots=True, repr=False)
class _Credentials:
    access_token: str
    refresh_token: str
    id_token: str | None
    email: str | None
    expires_at: int
    project_id: str = DEFAULT_PROJECT_ID

    @classmethod
    def from_tokens(
        cls,
        *,
        access_token: str,
        refresh_token: str,
        id_token: str | None = None,
        expires_in: int | None = None,
        project_id: str = DEFAULT_PROJECT_ID,
    ) -> _Credentials:
        email: str | None = None
        if id_token:
            email = _extract_email_from_jwt(id_token)

        now = int(time.time())
        expires_at = now + (expires_in if expires_in is not None else 3600)
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            email=email,
            expires_at=expires_at,
            project_id=project_id,
        )

    @classmethod
    def from_json(cls, payload: Any) -> _Credentials:
        if not isinstance(payload, dict):
            raise ValueError("Credential payload must be a dict")
        credentials = (
            payload.get("credentials") if "credentials" in payload else payload
        )
        if not isinstance(credentials, dict):
            raise ValueError("Credentials block must be an object")

        access_token = credentials.get("access_token")
        refresh_token = credentials.get("refresh_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("Missing access_token in credentials")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ValueError("Missing refresh_token in credentials")

        id_token = credentials.get("id_token")
        email = credentials.get("email")
        if not email and isinstance(id_token, str):
            email = _extract_email_from_jwt(id_token)

        expires_at = credentials.get("expires_at")
        if not isinstance(expires_at, int):
            expires_at = int(time.time()) + 3600

        project_id = credentials.get("project_id") or DEFAULT_PROJECT_ID

        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token if isinstance(id_token, str) else None,
            email=email if isinstance(email, str) else None,
            expires_at=expires_at,
            project_id=str(project_id),
        )

    @classmethod
    def from_cli_token_file(cls, path: Path) -> _Credentials | None:
        """Load from ~/.gemini/antigravity-cli/antigravity-oauth-token if present."""
        try:
            if not path.is_file():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            token = data.get("token")
            if not isinstance(token, dict):
                return None
            access_token = token.get("access_token")
            refresh_token = token.get("refresh_token")
            id_token = token.get("id_token")
            if not access_token or not refresh_token:
                return None

            # parse expiry timestamp (RFC3339 string or int)
            expiry_val = token.get("expiry")
            expires_at = int(time.time()) + 3600
            if isinstance(expiry_val, str):
                try:
                    dt = datetime.fromisoformat(expiry_val.replace("Z", "+00:00"))
                    expires_at = int(dt.timestamp())
                except Exception:
                    pass
            elif isinstance(expiry_val, int):
                expires_at = expiry_val

            email = _extract_email_from_jwt(id_token) if id_token else None

            return cls(
                access_token=access_token,
                refresh_token=refresh_token,
                id_token=id_token,
                email=email,
                expires_at=expires_at,
                project_id=DEFAULT_PROJECT_ID,
            )
        except Exception as exc:
            logger.debug("Failed to read native Antigravity CLI token: {}", exc)
            return None

    def as_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "credentials": {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "id_token": self.id_token,
                "email": self.email,
                "expires_at": self.expires_at,
                "project_id": self.project_id,
            },
        }


def _extract_email_from_jwt(id_token: str) -> str | None:
    try:
        parts = id_token.split(".")
        if len(parts) >= 2:
            payload_bytes = base64.urlsafe_b64decode(parts[1] + "===")
            payload = json.loads(payload_bytes)
            return payload.get("email")
    except Exception:
        pass
    return None


class AntigravityAuthManager:
    """Manage Antigravity credentials, interactive OAuth login, refresh, and revocation."""

    provider_id = "antigravity"

    def __init__(
        self,
        *,
        proxy: str | None = None,
        credential_path: Path | None = None,
        lock_path: Path | None = None,
        cli_token_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._credential_path = credential_path or antigravity_auth_path()
        self._lock_path = lock_path or antigravity_auth_lock_path()
        self._cli_token_path = cli_token_path or antigravity_cli_token_path()
        self._client = client or httpx.AsyncClient(
            proxy=proxy,
            timeout=httpx.Timeout(30.0),
        )
        self._owns_client = client is None
        self._lock = asyncio.Lock()
        self._credentials: _Credentials | None = None
        self._revision = 0
        self._last_error: str | None = None
        self._login_task: asyncio.Task[None] | None = None
        self._attempt_id: str | None = None
        self._authorization_url: str | None = None
        self._active_auth: BrowserAuthorization | None = None
        self._model_count: int = 0
        self._closed = False

        # Try to load existing credentials
        self._load_initial_credentials()

    def _load_initial_credentials(self) -> None:
        try:
            if self._credential_path.is_file():
                payload = json.loads(self._credential_path.read_text(encoding="utf-8"))
                self._credentials = _Credentials.from_json(payload)
                self._revision += 1
                return
        except Exception as exc:
            logger.warning("Could not load Antigravity credentials: {}", exc)

        # Fallback to CLI token file if present
        try:
            cli_creds = _Credentials.from_cli_token_file(self._cli_token_path)
            if cli_creds:
                self._credentials = cli_creds
                self._revision += 1
                # Save into FCC auth path
                with contextlib.suppress(Exception):
                    self._save_credentials_sync(cli_creds)
        except Exception as exc:
            logger.debug("Could not seed from native CLI token: {}", exc)

    def _save_credentials_sync(self, creds: _Credentials) -> None:
        self._write_credentials_unlocked(creds)

    def _write_credentials_unlocked(self, creds: _Credentials) -> None:
        self._credential_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            with contextlib.suppress(OSError):
                os.chmod(self._credential_path.parent, 0o700)
        temp_path = self._credential_path.with_name(
            f".{self._credential_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temp_path.write_text(
                json.dumps(creds.as_json(), indent=2), encoding="utf-8"
            )
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    os.chmod(temp_path, 0o600)
            os.replace(temp_path, self._credential_path)
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    os.chmod(self._credential_path, 0o600)
        finally:
            temp_path.unlink(missing_ok=True)

    def is_connected(self) -> bool:
        return self._credentials is not None and not self._closed

    def connected_provider_ids(self) -> tuple[str, ...]:
        return (self.provider_id,) if self.is_connected() else ()

    def set_model_count(self, count: int) -> None:
        self._model_count = count

    def status(self) -> ConnectedAccountStatus:
        connecting = self._login_task is not None and not self._login_task.done()
        state = (
            ConnectedAccountState.CONNECTING
            if connecting
            else ConnectedAccountState.ERROR
            if self._last_error
            else ConnectedAccountState.CONNECTED
            if self.is_connected()
            else ConnectedAccountState.DISCONNECTED
        )
        return ConnectedAccountStatus(
            provider_id=self.provider_id,
            state=state,
            connected=self.is_connected(),
            revision=self._revision,
            email=self._credentials.email if self._credentials else None,
            display_identity=self._credentials.email if self._credentials else None,
            supported_login_modes=(ConnectedAccountLoginMode.BROWSER,),
            default_login_mode=ConnectedAccountLoginMode.BROWSER,
            attempt_id=self._attempt_id if connecting else None,
            mode=ConnectedAccountLoginMode.BROWSER if connecting else None,
            authorization_url=self._authorization_url if connecting else None,
            model_count=self._model_count,
            message=self._last_error,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        async with self._lock:
            if self._login_task is not None and not self._login_task.done():
                return self.status()

            if mode != ConnectedAccountLoginMode.BROWSER:
                raise AntigravityLoginError(
                    "Antigravity only supports browser-based login."
                )

            auth = await BrowserAuthorization.start()
            self._active_auth = auth
            self._authorization_url = auth.auth_url
            self._attempt_id = uuid.uuid4().hex
            self._last_error = None
            self._login_task = asyncio.create_task(self._run_login(auth))
            return self.status()

    async def _run_login(self, auth: BrowserAuthorization) -> None:
        try:
            grant = await auth.wait()
            token_data = await exchange_authorization_code(self._client, grant)
            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token")
            if not access_token or not refresh_token:
                raise AntigravityLoginError("OAuth token exchange missing tokens.")

            id_token = token_data.get("id_token")
            expires_in = token_data.get("expires_in")

            creds = _Credentials.from_tokens(
                access_token=str(access_token),
                refresh_token=str(refresh_token),
                id_token=str(id_token) if id_token else None,
                expires_in=int(expires_in) if isinstance(expires_in, int) else 3600,
            )

            # Discover project ID from backend
            project_id = await self._discover_project(creds.access_token)
            if project_id:
                creds = _Credentials(
                    access_token=creds.access_token,
                    refresh_token=creds.refresh_token,
                    id_token=creds.id_token,
                    email=creds.email,
                    expires_at=creds.expires_at,
                    project_id=project_id,
                )

            async with self._lock:
                await self._save_credentials(creds)
                self._credentials = creds
                self._revision += 1
                self._last_error = None
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Antigravity login failed: {}", exc)
            async with self._lock:
                self._last_error = str(exc)
        finally:
            await auth.close()
            async with self._lock:
                self._active_auth = None
                self._authorization_url = None
                self._attempt_id = None

    async def cancel_login(self) -> ConnectedAccountStatus:
        auth_to_close, task_to_cancel = await self._cancel_login_internal()
        if auth_to_close is not None:
            await auth_to_close.close()
        if task_to_cancel is not None and not task_to_cancel.done():
            task_to_cancel.cancel()
        return self.status()

    async def _cancel_login_internal(
        self,
    ) -> tuple[BrowserAuthorization | None, asyncio.Task[None] | None]:
        async with self._lock:
            auth = self._active_auth
            task = self._login_task
            self._active_auth = None
            self._login_task = None
            self._authorization_url = None
            self._attempt_id = None
            self._last_error = None
            return auth, task

    async def disconnect(self) -> ConnectedAccountStatus:
        auth, task = await self._cancel_login_internal()
        if auth is not None:
            await auth.close()
        if task is not None and not task.done():
            task.cancel()

        async with self._lock:
            self._credentials = None
            self._revision += 1
            self._model_count = 0
            self._last_error = None

        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=10.0)
        if acquired:
            try:
                await asyncio.to_thread(self._credential_path.unlink, missing_ok=True)
            finally:
                await asyncio.to_thread(file_lock.release)
        else:
            self._credential_path.unlink(missing_ok=True)

        return self.status()

    async def access(self, *, force_refresh: bool = False) -> AntigravityAccess:
        """Return valid access token, auto-refreshing if expired or forced."""
        if self._closed:
            raise AntigravityReconnectRequired("Antigravity auth manager is closed.")

        async with self._lock:
            if self._credentials is None:
                raise AntigravityReconnectRequired(
                    "Antigravity is not connected. Sign in from the admin panel."
                )

            now = int(time.time())
            needs_refresh = force_refresh or (
                now >= self._credentials.expires_at - REFRESH_EARLY_SECONDS
            )
            if not needs_refresh:
                return AntigravityAccess(
                    access_token=self._credentials.access_token,
                    project_id=self._credentials.project_id,
                )

            # Perform token refresh
            refreshed = await self._refresh_token(self._credentials)
            self._credentials = refreshed
            await self._save_credentials(refreshed)
            self._revision += 1
            return AntigravityAccess(
                access_token=refreshed.access_token,
                project_id=refreshed.project_id,
            )

    async def _refresh_token(self, creds: _Credentials) -> _Credentials:
        payload = {
            "grant_type": "refresh_token",
            "client_id": ANTIGRAVITY_CLIENT_ID,
            "client_secret": ANTIGRAVITY_CLIENT_SECRET,
            "refresh_token": creds.refresh_token,
        }
        response = await self._client.post(
            GOOGLE_OAUTH_TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not response.is_success:
            detail = response.text
            try:
                err = response.json()
                detail = err.get("error_description") or err.get("error") or detail
            except Exception:
                pass
            raise AntigravityReconnectRequired(
                f"Antigravity token refresh failed: {detail}"
            )

        data = response.json()
        new_access_token = data.get("access_token")
        if not new_access_token or not isinstance(new_access_token, str):
            raise AntigravityReconnectRequired(
                "Malformed refresh response from Google OAuth."
            )

        expires_in = data.get("expires_in")
        new_expires_at = int(time.time()) + (
            int(expires_in) if isinstance(expires_in, int) else 3600
        )
        new_id_token = data.get("id_token") or creds.id_token
        new_refresh_token = data.get("refresh_token") or creds.refresh_token

        return _Credentials(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            id_token=str(new_id_token) if new_id_token else None,
            email=creds.email,
            expires_at=new_expires_at,
            project_id=creds.project_id,
        )

    async def _discover_project(self, access_token: str) -> str:
        """Query loadCodeAssist to discover cloudaicompanionProject."""
        for host in (
            "daily-cloudcode-pa.googleapis.com",
            "cloudcode-pa.googleapis.com",
        ):
            url = f"https://{host}/v1internal:loadCodeAssist"
            try:
                resp = await self._client.post(
                    url,
                    json={},
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "User-Agent": "Antigravity",
                    },
                )
                if resp.is_success:
                    body = resp.json()
                    proj = body.get("cloudaicompanionProject")
                    if proj and isinstance(proj, str):
                        return proj
            except Exception as exc:
                logger.debug("Failed to discover project from {}: {}", host, exc)
        return DEFAULT_PROJECT_ID

    async def _save_credentials(self, creds: _Credentials) -> None:
        file_lock = InterprocessFileLock(self._lock_path)
        acquired = await asyncio.to_thread(file_lock.acquire, wait=True, timeout=10.0)
        if not acquired:
            raise AntigravityLoginError(
                "Could not lock the Antigravity credential file."
            )
        try:
            await asyncio.to_thread(self._write_credentials_unlocked, creds)
        finally:
            await asyncio.to_thread(file_lock.release)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        auth, task = await self._cancel_login_internal()
        if auth is not None:
            await auth.close()
        if task is not None and not task.done():
            task.cancel()
        if self._owns_client:
            await self._client.aclose()
