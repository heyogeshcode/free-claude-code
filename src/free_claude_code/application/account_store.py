"""Multi-Account hierarchical store with atomic persistence and model-level cooldowns."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from free_claude_code.config.paths import (
    accounts_config_path,
    accounts_lock_path,
    antigravity_auth_path,
    github_copilot_auth_path,
    openai_auth_path,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

current_account_id: ContextVar[str | None] = ContextVar(
    "current_account_id", default=None
)
_GLOBAL_ACCOUNT_STORE: AccountStore | None = None


def get_account_store() -> AccountStore:
    global _GLOBAL_ACCOUNT_STORE
    if _GLOBAL_ACCOUNT_STORE is None:
        _GLOBAL_ACCOUNT_STORE = AccountStore()
    return _GLOBAL_ACCOUNT_STORE


@dataclass
class Account:
    """One authenticated provider account."""

    id: str
    label: str
    priority: int
    status: str = "ACTIVE"  # ACTIVE, DISABLED, RATE_LIMITED
    credentials: dict[str, Any] = field(default_factory=dict)
    model_cooldowns: dict[str, float] = field(
        default_factory=dict
    )  # model -> unix timestamp

    def is_active(self) -> bool:
        return self.status.upper() == "ACTIVE"

    def is_cooling(self, model: str) -> bool:
        """Return True if this account is currently cooling down for the given model."""
        now = time.time()
        expiry = self.model_cooldowns.get(model, 0.0)
        return expiry > now

    def cooldown_remaining(self, model: str) -> float:
        """Remaining seconds on cooldown for model, or 0.0."""
        expiry = self.model_cooldowns.get(model, 0.0)
        remaining = expiry - time.time()
        return max(0.0, remaining)

    def set_cooldown(self, model: str, duration_seconds: float) -> None:
        """Set a cooldown on this account for a specific model."""
        self.model_cooldowns[model] = time.time() + max(1.0, duration_seconds)

    def clear_cooldown(self, model: str) -> None:
        self.model_cooldowns.pop(model, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "priority": self.priority,
            "status": self.status,
            "credentials": self.credentials,
            "model_cooldowns": self.model_cooldowns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Account:
        return cls(
            id=str(data.get("id", "")),
            label=str(data.get("label", "")),
            priority=int(data.get("priority", 1)),
            status=str(data.get("status", "ACTIVE")),
            credentials=dict(data.get("credentials") or {}),
            model_cooldowns={
                k: float(v) for k, v in (data.get("model_cooldowns") or {}).items()
            },
        )


@dataclass
class ProviderPool:
    """Collection of accounts for a single provider."""

    strategy: str = "model_preserving_priority"
    accounts: list[Account] = field(default_factory=list)

    def sorted_accounts(self) -> list[Account]:
        """Return accounts sorted by priority ascending (1 = highest priority)."""
        return sorted(self.accounts, key=lambda a: a.priority)

    def normalize_priorities(self) -> None:
        """Ensure priorities are strictly 1, 2, ..., N."""
        sorted_accs = sorted(self.accounts, key=lambda a: a.priority)
        for idx, acc in enumerate(sorted_accs, start=1):
            acc.priority = idx


def _extract_email_from_creds(creds: Any) -> str | None:
    if not isinstance(creds, dict):
        return None
    inner = (
        creds.get("credentials")
        if isinstance(creds.get("credentials"), dict)
        else creds
    )
    if inner.get("email"):
        return str(inner["email"])
    id_token = inner.get("id_token") or creds.get("id_token")
    if isinstance(id_token, str) and "." in id_token:
        try:
            import base64

            parts = id_token.split(".")
            if len(parts) >= 2:
                padding = "=" * (4 - len(parts[1]) % 4)
                data = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
                if data.get("email"):
                    return str(data["email"])
                profile = data.get("https://api.openai.com/profile")
                if isinstance(profile, dict) and profile.get("email"):
                    return str(profile["email"])
        except Exception:
            pass
    return None


class AccountStore:
    """Persistent, thread-safe, and process-safe store for multi-account management."""

    @staticmethod
    def canonical_provider(provider_id: str) -> str:
        mapping = {
            "chatgpt": "openai",
            "openai": "openai",
            "copilot": "github_copilot",
            "github_copilot": "github_copilot",
            "antigravity": "antigravity",
        }
        return mapping.get(provider_id.lower(), provider_id.lower())

    def __init__(
        self,
        config_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        self._path = config_path or accounts_config_path()
        self._lock_path = lock_path or accounts_lock_path()
        self._async_lock = asyncio.Lock()
        self._providers: dict[str, ProviderPool] = {
            "antigravity": ProviderPool(),
            "openai": ProviderPool(),
            "github_copilot": ProviderPool(),
        }
        self.load()

    def get_providers(self) -> list[str]:
        return list(self._providers.keys())

    def get_accounts(self, provider_id: str) -> list[Account]:
        """Return accounts for provider sorted by priority ascending."""
        p_id = self.canonical_provider(provider_id)
        pool = self._providers.get(p_id)
        if pool is None:
            return []
        return pool.sorted_accounts()

    def get_account_by_id(self, provider_id: str, account_id: str) -> Account | None:
        p_id = self.canonical_provider(provider_id)
        pool = self._providers.get(p_id)
        if pool is None:
            return None
        for acc in pool.accounts:
            if acc.id == account_id:
                return acc
        return None

    def get_healthy_accounts_for_model(
        self, provider_id: str, model: str
    ) -> list[Account]:
        """Return active accounts not in cooldown for model, sorted by priority ascending."""
        accounts = self.get_accounts(provider_id)
        return [
            acc for acc in accounts if acc.is_active() and not acc.is_cooling(model)
        ]

    def set_model_cooldown(
        self,
        provider_id: str,
        account_id: str,
        model: str,
        duration_seconds: float = 60.0,
    ) -> bool:
        """Set a cooldown on a specific account for a given model and persist atomically."""
        p_id = self.canonical_provider(provider_id)
        acc = self.get_account_by_id(p_id, account_id)
        if acc is None:
            return False
        acc.set_cooldown(model, duration_seconds)
        logger.info(
            "Account cooldown set: provider='{}' account='{}' model='{}' duration={}s",
            p_id,
            account_id,
            model,
            duration_seconds,
        )
        self.save()
        return True

    def add_account(
        self,
        provider_id: str,
        label: str,
        credentials: dict[str, Any],
        account_id: str | None = None,
        priority: int | None = None,
        status: str = "ACTIVE",
    ) -> Account:
        """Add a new account to the provider pool and persist."""
        p_id = self.canonical_provider(provider_id)
        if p_id not in self._providers:
            self._providers[p_id] = ProviderPool()
        pool = self._providers[p_id]

        if not account_id:
            account_id = f"{p_id}_acc_{len(pool.accounts) + 1:02d}"

        # If priority not specified, append to end
        if priority is None:
            priority = len(pool.accounts) + 1

        acc = Account(
            id=account_id,
            label=label,
            priority=priority,
            status=status,
            credentials=credentials,
        )
        # Check if ID already exists, replace or append
        existing = [a for a in pool.accounts if a.id == account_id]
        if existing:
            pool.accounts.remove(existing[0])
        pool.accounts.append(acc)
        pool.normalize_priorities()
        self.save()
        return acc

    def update_account(
        self,
        provider_id: str,
        account_id: str,
        label: str | None = None,
        status: str | None = None,
        credentials: dict[str, Any] | None = None,
        priority: int | None = None,
    ) -> Account | None:
        """Update account attributes and persist."""
        p_id = self.canonical_provider(provider_id)
        acc = self.get_account_by_id(p_id, account_id)
        if acc is None:
            return None
        if label is not None:
            acc.label = label
        if status is not None:
            acc.status = status
        if credentials is not None:
            acc.credentials.update(credentials)
        if priority is not None:
            acc.priority = priority
            pool = self._providers[p_id]
            pool.normalize_priorities()
        self.save()
        return acc

    def delete_account(self, provider_id: str, account_id: str) -> bool:
        """Remove an account and normalize remaining priorities."""
        p_id = self.canonical_provider(provider_id)
        pool = self._providers.get(p_id)
        if pool is None:
            return False
        target = self.get_account_by_id(p_id, account_id)
        if target is None:
            return False
        pool.accounts.remove(target)
        pool.normalize_priorities()
        self.save()
        return True

    def move_up(self, provider_id: str, account_id: str) -> bool:
        """Move account up in priority (e.g. from index 3 to index 2)."""
        p_id = self.canonical_provider(provider_id)
        pool = self._providers.get(p_id)
        if pool is None:
            return False
        sorted_accs = pool.sorted_accounts()
        target_idx = -1
        for idx, acc in enumerate(sorted_accs):
            if acc.id == account_id:
                target_idx = idx
                break
        if target_idx <= 0:
            return False  # Already top or not found

        # Swap with preceding account
        sorted_accs[target_idx - 1], sorted_accs[target_idx] = (
            sorted_accs[target_idx],
            sorted_accs[target_idx - 1],
        )
        for idx, acc in enumerate(sorted_accs, start=1):
            acc.priority = idx
        pool.accounts = sorted_accs
        self.save()
        return True

    def move_down(self, provider_id: str, account_id: str) -> bool:
        """Move account down in priority (e.g. from index 1 to index 2)."""
        p_id = self.canonical_provider(provider_id)
        pool = self._providers.get(p_id)
        if pool is None:
            return False
        sorted_accs = pool.sorted_accounts()
        target_idx = -1
        for idx, acc in enumerate(sorted_accs):
            if acc.id == account_id:
                target_idx = idx
                break
        if target_idx < 0 or target_idx >= len(sorted_accs) - 1:
            return False  # Already bottom or not found

        # Swap with following account
        sorted_accs[target_idx], sorted_accs[target_idx + 1] = (
            sorted_accs[target_idx + 1],
            sorted_accs[target_idx],
        )
        for idx, acc in enumerate(sorted_accs, start=1):
            acc.priority = idx
        pool.accounts = sorted_accs
        self.save()
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "2.0.0",
            "providers": {
                p_id: {
                    "strategy": pool.strategy,
                    "accounts": [a.to_dict() for a in pool.sorted_accounts()],
                }
                for p_id, pool in self._providers.items()
            },
        }

    def load(self) -> None:
        """Load from disk or auto-migrate legacy credential files."""
        if not self._path.exists():
            self._auto_migrate_legacy()
            return

        try:
            with InterprocessFileLock(self._lock_path):
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
            self._parse_data(data)
            self._sync_single_credential_files()
        except Exception as exc:
            logger.warning(
                "Could not load account configuration from {}: {}. Attempting migration.",
                self._path,
                exc,
            )
            self._auto_migrate_legacy()

    def _sync_single_credential_files(self) -> None:
        """Synchronize accounts from single-auth credential files if missing from store."""
        changed = False

        # Upgrade any legacy non-email labels if email is discoverable in credentials
        for p in ("openai", "antigravity"):
            for acc in self._providers[p].accounts:
                if "@" not in acc.label:
                    acc_email = _extract_email_from_creds(acc.credentials)
                    if acc_email:
                        acc.label = acc_email
                        changed = True

        # 1. Sync OpenAI credentials file if present
        oa_path = openai_auth_path()
        if oa_path.exists():
            try:
                with open(oa_path, encoding="utf-8") as f:
                    data = json.load(f)
                creds = data.get("credentials") or data
                if isinstance(creds, dict) and (creds.get("access_token") or creds.get("refresh_token")):
                    email = _extract_email_from_creds(creds)
                    oa_accounts = self._providers["openai"].accounts
                    existing = False
                    for acc in oa_accounts:
                        acc_email = _extract_email_from_creds(acc.credentials)
                        if email and acc_email and email.lower() == acc_email.lower():
                            existing = True
                            acc.credentials.update(creds)
                            if email and "@" not in acc.label:
                                acc.label = email
                            break
                        if acc.credentials.get("account_id") and creds.get("account_id") and acc.credentials.get("account_id") == creds.get("account_id"):
                            existing = True
                            acc.credentials.update(creds)
                            if email and "@" not in acc.label:
                                acc.label = email
                            break
                    if not existing:
                        label = email or "OpenAI / ChatGPT"
                        acc_id = f"openai_{email}" if email else f"openai_acc_{len(oa_accounts) + 1:02d}"
                        self._providers["openai"].accounts.append(
                            Account(
                                id=acc_id,
                                label=label,
                                priority=len(oa_accounts) + 1,
                                status="ACTIVE",
                                credentials=creds,
                            )
                        )
                        self._providers["openai"].normalize_priorities()
                        changed = True
            except Exception as exc:
                logger.debug("Failed syncing single OpenAI credentials: {}", exc)

        # 2. Sync Antigravity credentials file if present
        ag_path = antigravity_auth_path()
        if ag_path.exists():
            try:
                with open(ag_path, encoding="utf-8") as f:
                    data = json.load(f)
                creds = data.get("credentials") or data
                if isinstance(creds, dict) and (creds.get("access_token") or creds.get("refresh_token")):
                    email = _extract_email_from_creds(creds)
                    ag_accounts = self._providers["antigravity"].accounts
                    existing = False
                    for acc in ag_accounts:
                        acc_email = _extract_email_from_creds(acc.credentials)
                        if email and acc_email and email.lower() == acc_email.lower():
                            existing = True
                            acc.credentials.update(creds)
                            if email and "@" not in acc.label:
                                acc.label = email
                            break
                    if not existing:
                        label = email or "Antigravity"
                        acc_id = f"ag_{email}" if email else f"ag_acc_{len(ag_accounts) + 1:02d}"
                        self._providers["antigravity"].accounts.append(
                            Account(
                                id=acc_id,
                                label=label,
                                priority=len(ag_accounts) + 1,
                                status="ACTIVE",
                                credentials=creds,
                            )
                        )
                        self._providers["antigravity"].normalize_priorities()
                        changed = True
            except Exception as exc:
                logger.debug("Failed syncing single Antigravity credentials: {}", exc)

        if changed:
            self.save()

    def _parse_data(self, data: dict[str, Any]) -> None:
        providers_dict = data.get("providers", {})
        for p_id, p_data in providers_dict.items():
            canon_id = self.canonical_provider(p_id)
            strategy = p_data.get("strategy", "model_preserving_priority")
            accounts = [
                Account.from_dict(acc_data) for acc_data in p_data.get("accounts", [])
            ]
            pool = self._providers.get(canon_id)
            if pool is None:
                pool = ProviderPool(strategy=strategy, accounts=accounts)
                self._providers[canon_id] = pool
            else:
                pool.strategy = strategy
                for acc in accounts:
                    if not any(a.id == acc.id for a in pool.accounts):
                        pool.accounts.append(acc)
            pool.normalize_priorities()

        for req_p in ("antigravity", "openai", "github_copilot"):
            if req_p not in self._providers:
                self._providers[req_p] = ProviderPool()

    def _auto_migrate_legacy(self) -> None:
        """Seed accounts from legacy single-account files if available."""
        migrated = False

        # 1. Antigravity
        ag_path = antigravity_auth_path()
        if ag_path.exists() and not self._providers["antigravity"].accounts:
            try:
                with open(ag_path, encoding="utf-8") as f:
                    data = json.load(f)
                creds = data.get("credentials") or data
                email = _extract_email_from_creds(creds)
                label = email or "Primary Antigravity Account"
                acc_id = f"ag_{email}" if email else "antigravity_acc_01"
                self._providers["antigravity"].accounts.append(
                    Account(
                        id=acc_id,
                        label=label,
                        priority=1,
                        status="ACTIVE",
                        credentials=creds,
                    )
                )
                migrated = True
            except Exception as exc:
                logger.debug("Legacy Antigravity migration skipped: {}", exc)

        # 2. ChatGPT / OpenAI
        oa_path = openai_auth_path()
        if oa_path.exists() and not self._providers["openai"].accounts:
            try:
                with open(oa_path, encoding="utf-8") as f:
                    data = json.load(f)
                creds = data.get("credentials") or data
                email = _extract_email_from_creds(creds)
                label = email or "Primary OpenAI / ChatGPT Account"
                acc_id = f"openai_{email}" if email else "openai_acc_01"
                self._providers["openai"].accounts.append(
                    Account(
                        id=acc_id,
                        label=label,
                        priority=1,
                        status="ACTIVE",
                        credentials=creds,
                    )
                )
                migrated = True
            except Exception as exc:
                logger.debug("Legacy ChatGPT migration skipped: {}", exc)

        # 3. Copilot
        gh_path = github_copilot_auth_path()
        if gh_path.exists() and not self._providers["github_copilot"].accounts:
            try:
                with open(gh_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._providers["github_copilot"].accounts.append(
                    Account(
                        id="copilot_acc_01",
                        label="Primary Copilot Account",
                        priority=1,
                        status="ACTIVE",
                        credentials=data,
                    )
                )
                migrated = True
            except Exception as exc:
                logger.debug("Legacy Copilot migration skipped: {}", exc)

        if migrated:
            self.save()

    def save(self) -> None:
        """Atomically persist state to disk via temporary file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        serialized = json.dumps(payload, indent=2)

        with InterprocessFileLock(self._lock_path):
            dir_path = self._path.parent
            fd, tmp_path = tempfile.mkstemp(
                prefix="accounts_",
                suffix=".tmp",
                dir=dir_path,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(serialized)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
