"""NVIDIA NIM dynamic API key discovery, sanitation, health tracking, and persistence."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from free_claude_code.config.paths import (
    config_dir_path,
    nvidia_key_state_path,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
BASE_BACKOFF_SECONDS = 60.0
REPEAT_LIMIT_WINDOW_SECONDS = 300.0  # 5 minutes


@dataclass
class KeyHealth:
    """Health statistics and lifecycle state for one API key."""

    state: str = "HEALTHY"  # HEALTHY, RATE_LIMITED, DEAD
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    consecutive_hangs: int = 0
    last_latency_ms: float = 0.0
    latency_ema_ms: float = 0.0
    cooldown_until: float = 0.0
    total_requests: int = 0
    last_rate_limited_at: float = 0.0
    current_backoff_seconds: float = BASE_BACKOFF_SECONDS

    def is_available(self, now: float | None = None) -> bool:
        if self.state == "DEAD":
            return False
        current_time = now if now is not None else time.time()
        return self.cooldown_until <= current_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_hangs": self.consecutive_hangs,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "latency_ema_ms": round(self.latency_ema_ms, 2),
            "cooldown_until": round(self.cooldown_until, 2),
            "total_requests": self.total_requests,
            "last_rate_limited_at": round(self.last_rate_limited_at, 2),
            "current_backoff_seconds": round(self.current_backoff_seconds, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyHealth:
        return cls(
            state=str(data.get("state", "HEALTHY")),
            consecutive_successes=int(data.get("consecutive_successes", 0)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            consecutive_hangs=int(data.get("consecutive_hangs", 0)),
            last_latency_ms=float(data.get("last_latency_ms", 0.0)),
            latency_ema_ms=float(data.get("latency_ema_ms", 0.0)),
            cooldown_until=float(data.get("cooldown_until", 0.0)),
            total_requests=int(data.get("total_requests", 0)),
            last_rate_limited_at=float(data.get("last_rate_limited_at", 0.0)),
            current_backoff_seconds=float(
                data.get("current_backoff_seconds", BASE_BACKOFF_SECONDS)
            ),
        )


class NvidiaKeyStore:
    """Manages discovery, filtering, validation, and persistent health for NVIDIA keys."""

    def __init__(
        self,
        key_file: Path | str | None = None,
        state_file: Path | str | None = None,
        auto_shield_git: bool = True,
    ) -> None:
        self._key_file = Path(key_file) if key_file else self.find_key_file()
        self._state_file = Path(state_file) if state_file else nvidia_key_state_path()
        self._lock_path = self._state_file.with_suffix(".lock")
        self._keys: list[str] = []
        self._key_health: dict[str, KeyHealth] = {}
        self._last_save_time = 0.0

        if auto_shield_git:
            self.ensure_git_shielding()

        self.reload_keys()
        self.load_state()

    @staticmethod
    def find_key_file() -> Path | None:
        """Search order: ENV, CWD, repo root, ~/.free-claude-code, ~/.fcc, user home."""
        # 1. Environment variable override
        env_override = os.environ.get("NVIDIA_KEYS_FILE")
        if env_override:
            p = Path(env_override)
            if p.is_file():
                return p

        candidates = [
            Path(__file__).resolve().parents[3] / "nvidia_working.txt",
            Path("/home/heyogesh/free-claude-code/nvidia_working.txt"),
            Path.cwd() / "nvidia_working.txt",
            Path(__file__).resolve().parents[4] / "nvidia_working.txt",
            Path.home() / ".free-claude-code" / "nvidia_working.txt",
            config_dir_path() / "nvidia_working.txt",
            Path.home() / "nvidia_working.txt",
            Path.home()
            / "Documents"
            / "heyogesh_coding"
            / "Harvester"
            / "nvidia"
            / "nvidia_working.txt",
        ]
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate
            except (OSError, PermissionError):
                continue
        return None

    @staticmethod
    def ensure_git_shielding(repo_root: Path | None = None) -> None:
        """Ensure sensitive NVIDIA key files and state files are in .gitignore."""
        if repo_root is None:
            # Look up hierarchy from this file
            cur = Path(__file__).resolve().parent
            while cur != cur.parent:
                if (cur / ".git").is_dir():
                    repo_root = cur
                    break
                cur = cur.parent

        if not repo_root:
            return

        gitignore = repo_root / ".gitignore"
        patterns_to_ensure = [
            "nvidia_working.txt",
            "*.state.json",
            ".nvidia_*",
            ".nvidia_key_state.json",
            "accounts.json",
            ".accounts.json",
        ]

        existing_lines: set[str] = set()
        if gitignore.is_file():
            with contextlib.suppress(Exception):
                existing_lines = {
                    line.strip()
                    for line in gitignore.read_text(encoding="utf-8").splitlines()
                }

        missing = [p for p in patterns_to_ensure if p not in existing_lines]
        if missing:
            try:
                with open(gitignore, "a", encoding="utf-8") as f:
                    f.write("\n# Nvidia & Account Credential Safeguards\n")
                    for p in missing:
                        f.write(f"{p}\n")
                logger.info("Updated .gitignore with security safeguards: {}", missing)
            except Exception as exc:
                logger.warning("Failed to update .gitignore: {}", exc)

    @staticmethod
    def sanitize_line(raw_line: str) -> str | None:
        """Clean and validate one key line."""
        # Strip ANSI codes
        cleaned = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        # Discard comments and empty lines
        if not cleaned or cleaned.startswith("#") or cleaned.startswith("//"):
            return None

        # Strip surrounding quotes, trailing commas, trailing semicolons
        cleaned = cleaned.strip("\"' ,;")
        if not cleaned or cleaned.startswith("#"):
            return None

        # Validate basic API key format (NVIDIA keys typically nvapi-... or alphanumeric >= 20)
        if len(cleaned) < 20 or " " in cleaned:
            return None

        return cleaned

    def reload_keys(self) -> list[str]:
        """Read and sanitize keys from discovery file."""
        if not self._key_file or not self._key_file.is_file():
            logger.debug("No nvidia_working.txt file found during reload_keys.")
            self._keys = []
            return []

        discovered: list[str] = []
        seen: set[str] = set()
        try:
            with open(self._key_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    sanitized = self.sanitize_line(line)
                    if sanitized and sanitized not in seen:
                        seen.add(sanitized)
                        discovered.append(sanitized)
        except Exception as exc:
            logger.error("Error reading NVIDIA key file {}: {}", self._key_file, exc)

        self._keys = discovered
        logger.info(
            "NvidiaKeyStore loaded {} sanitized keys from {}",
            len(self._keys),
            self._key_file,
        )
        return self._keys

    def discover_and_load(self) -> list[str]:
        """Alias for reload_keys."""
        return self.reload_keys()

    def get_all_keys(self) -> list[str]:
        return list(self._keys)

    def get_healthy_keys(self) -> list[str]:
        """Return available keys sorted by latency EMA and reliability."""
        now = time.time()
        available: list[str] = []
        for key in self._keys:
            health = self._get_or_create_health(key)
            if health.is_available(now):
                available.append(key)

        # Sort: fewer hangs first, then by latency EMA (0 means unmeasured, placed in middle)
        def sort_key(k: str) -> tuple[int, float]:
            h = self._key_health[k]
            # Prioritize low hangs, and measured low latency over unmeasured
            latency_score = h.latency_ema_ms if h.latency_ema_ms > 0 else 500.0
            return (h.consecutive_hangs, latency_score)

        return sorted(available, key=sort_key)

    def get_primary_key(self) -> str | None:
        healthy = self.get_healthy_keys()
        if healthy:
            return healthy[0]
        # Fallback to any non-dead key if all are temporarily rate-limited
        for key in self._keys:
            if self._key_health.get(key, KeyHealth()).state != "DEAD":
                return key
        return self._keys[0] if self._keys else None

    def _get_or_create_health(self, key: str) -> KeyHealth:
        if key not in self._key_health:
            self._key_health[key] = KeyHealth()
        return self._key_health[key]

    def get_key_health(self, key: str) -> KeyHealth | None:
        """Return KeyHealth for a key if tracked."""
        return self._key_health.get(key)

    def record_success(self, key: str, latency_ms: float) -> None:
        """Mark key healthy, update latency EMA, reset failures/hangs."""
        health = self._get_or_create_health(key)
        health.state = "HEALTHY"
        health.consecutive_successes += 1
        health.consecutive_failures = 0
        health.consecutive_hangs = 0
        health.cooldown_until = 0.0
        health.total_requests += 1
        health.last_latency_ms = latency_ms

        if health.latency_ema_ms <= 0:
            health.latency_ema_ms = latency_ms
        else:
            # Exponential moving average with alpha=0.3
            health.latency_ema_ms = (0.7 * health.latency_ema_ms) + (0.3 * latency_ms)

        self._maybe_save_state()

    def record_rate_limited(
        self, key: str, backoff_seconds: float = BASE_BACKOFF_SECONDS
    ) -> None:
        """Handle 429 response with exponential penalty on repeat offenses."""
        now = time.time()
        health = self._get_or_create_health(key)
        health.state = "RATE_LIMITED"
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.total_requests += 1

        # Exponential backoff if repeated within window
        if (now - health.last_rate_limited_at) <= REPEAT_LIMIT_WINDOW_SECONDS:
            health.current_backoff_seconds = min(
                3600.0, health.current_backoff_seconds * 2
            )
        else:
            health.current_backoff_seconds = backoff_seconds

        health.last_rate_limited_at = now
        health.cooldown_until = now + health.current_backoff_seconds
        logger.warning(
            "NVIDIA key {} rate-limited. Cooldown for {}s (until {})",
            key[:12] + "...",
            health.current_backoff_seconds,
            health.cooldown_until,
        )
        self.save_state()

    def record_hang(self, key: str) -> None:
        """Record a socket timeout (>15s without first token)."""
        now = time.time()
        health = self._get_or_create_health(key)
        health.consecutive_hangs += 1
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.total_requests += 1
        # Place on brief 30s cooldown to deprioritize
        health.cooldown_until = now + 30.0
        logger.warning(
            "NVIDIA key {} timed out/hung. Consecutive hangs: {}",
            key[:12] + "...",
            health.consecutive_hangs,
        )
        self._maybe_save_state()

    def record_dead(self, key: str, reason: str = "401/403 Invalid Key") -> None:
        """Permanently exclude dead/unauthorized key."""
        health = self._get_or_create_health(key)
        health.state = "DEAD"
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.cooldown_until = 9999999999.0
        health.total_requests += 1
        logger.error("NVIDIA key {} marked DEAD (reason: {})", key[:12] + "...", reason)
        self.save_state()

    def _maybe_save_state(self) -> None:
        """Debounce state file saves to at most once every 5 seconds."""
        now = time.time()
        if now - self._last_save_time >= 5.0:
            self.save_state()

    def save_state(self) -> None:
        """Persist health ledger atomically to .nvidia_key_state.json."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": int(time.time()),
            "keys": {k: h.to_dict() for k, h in self._key_health.items()},
        }
        serialized = json.dumps(data, indent=2)
        try:
            with InterprocessFileLock(self._lock_path):
                dir_path = self._state_file.parent
                fd, tmp_path = tempfile.mkstemp(
                    prefix="nvidia_state_",
                    suffix=".tmp",
                    dir=dir_path,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(serialized)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self._state_file)
                    self._last_save_time = time.time()
                except Exception:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
        except Exception as exc:
            logger.warning("Failed to save NVIDIA key state ledger: {}", exc)

    def load_state(self) -> None:
        """Load state ledger from disk."""
        if not self._state_file.is_file():
            return
        try:
            with InterprocessFileLock(self._lock_path):
                with open(self._state_file, encoding="utf-8") as f:
                    data = json.load(f)
                keys_data = data.get("keys", {})
                for k, h_data in keys_data.items():
                    if isinstance(h_data, dict):
                        self._key_health[k] = KeyHealth.from_dict(h_data)
            logger.info(
                "Loaded health history for {} NVIDIA keys from {}",
                len(self._key_health),
                self._state_file,
            )
        except Exception as exc:
            logger.warning(
                "Could not read NVIDIA key state from {}: {}", self._state_file, exc
            )

    def record_rate_limit(
        self, key: str, backoff_seconds: float = BASE_BACKOFF_SECONDS
    ) -> None:
        """Alias for record_rate_limited."""
        self.record_rate_limited(key, backoff_seconds=backoff_seconds)

    def record_failure(
        self, key: str, unrecoverable: bool = False, reason: str = "Request failed"
    ) -> None:
        """Record a failure; if unrecoverable, mark key DEAD."""
        if unrecoverable:
            self.record_dead(key, reason=reason)
            return
        health = self._get_or_create_health(key)
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.total_requests += 1
        self._maybe_save_state()


_GLOBAL_NVIDIA_KEY_STORE: NvidiaKeyStore | None = None


def get_nvidia_key_store() -> NvidiaKeyStore:
    """Get or create singleton NvidiaKeyStore."""
    global _GLOBAL_NVIDIA_KEY_STORE
    if _GLOBAL_NVIDIA_KEY_STORE is None:
        store = NvidiaKeyStore()
        store.discover_and_load()
        _GLOBAL_NVIDIA_KEY_STORE = store
    return _GLOBAL_NVIDIA_KEY_STORE
