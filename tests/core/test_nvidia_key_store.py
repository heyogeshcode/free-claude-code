"""Unit tests for NvidiaKeyStore."""

from pathlib import Path

from free_claude_code.core.nvidia_key_store import NvidiaKeyStore


def test_nvidia_key_store_sanitation_and_health(tmp_path: Path) -> None:
    raw_keys = """
    # Comment line
    "nvapi-VALID_KEY_ONE_12345678901234567890",
    nvapi-VALID_KEY_TWO_12345678901234567890;
    // Another comment
    short
    \x1b[31mnvapi-VALID_KEY_THREE_12345678901234567890\x1b[0m
    "nvapi-VALID_KEY_ONE_12345678901234567890"  # Duplicate
    """
    key_file = tmp_path / "nvidia_working.txt"
    key_file.write_text(raw_keys, encoding="utf-8")
    state_file = tmp_path / ".nvidia_key_state.json"

    store = NvidiaKeyStore(
        key_file=key_file, state_file=state_file, auto_shield_git=False
    )
    keys = store.get_all_keys()

    assert len(keys) == 3
    assert keys[0] == "nvapi-VALID_KEY_ONE_12345678901234567890"
    assert keys[1] == "nvapi-VALID_KEY_TWO_12345678901234567890"
    assert keys[2] == "nvapi-VALID_KEY_THREE_12345678901234567890"

    # All healthy initially
    healthy = store.get_healthy_keys()
    assert len(healthy) == 3

    # Record rate limited on key 1
    store.record_rate_limited(keys[0], backoff_seconds=120)
    healthy_after_limit = store.get_healthy_keys()
    assert keys[0] not in healthy_after_limit
    assert len(healthy_after_limit) == 2

    # Record dead on key 2
    store.record_dead(keys[1], reason="401 Unauthorized")
    healthy_after_dead = store.get_healthy_keys()
    assert healthy_after_dead == [keys[2]]

    # Record success and latency EMA on key 3
    store.record_success(keys[2], latency_ms=250.0)
    store.record_success(keys[2], latency_ms=150.0)
    assert store.get_primary_key() == keys[2]

    # Reload state from disk
    store.save_state()
    store2 = NvidiaKeyStore(
        key_file=key_file, state_file=state_file, auto_shield_git=False
    )
    assert store2.get_healthy_keys() == [keys[2]]
