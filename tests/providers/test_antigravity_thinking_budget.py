"""Tests for Phase 2: Thinking/Reasoning Budget Fix in Antigravity provider."""

from free_claude_code.core.anthropic.models import MessagesRequest, ThinkingConfig
from free_claude_code.providers.antigravity.translator import translate_messages_request


def test_thinking_with_pydantic_model():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Solve this puzzle"}],
        max_tokens=8192,
        thinking=ThinkingConfig(type="enabled", budget_tokens=4096),
    )
    cache: dict[str, str] = {}
    payload, backend_model = translate_messages_request(request, "proj-1", cache)

    assert backend_model == "gemini-3.8-flash-tiered"
    gen_config = payload["request"]["generationConfig"]
    assert "thinkingConfig" in gen_config
    assert gen_config["thinkingConfig"]["thinkingBudget"] == 4096
    # Gemini budget splitting: 8192 - 4096 = 4096
    assert gen_config["maxOutputTokens"] == 4096


def test_thinking_with_dict():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Compute"}],
        max_tokens=32768,
        thinking={"type": "enabled", "budget_tokens": 16384},
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "proj-1", cache)

    gen_config = payload["request"]["generationConfig"]
    assert gen_config["thinkingConfig"]["thinkingBudget"] == 16384
    # Gemini budget splitting: 32768 - 16384 = 16384
    assert gen_config["maxOutputTokens"] == 16384


def test_gemini_budget_splitting_small_margin_floor():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Quick answer"}],
        max_tokens=2500,
        thinking=ThinkingConfig(type="enabled", budget_tokens=2000),
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "proj-1", cache)

    gen_config = payload["request"]["generationConfig"]
    assert gen_config["thinkingConfig"]["thinkingBudget"] == 2000
    # 2500 - 2000 = 500, which is below 1024 floor, so it should be clamped to 1024
    assert gen_config["maxOutputTokens"] == 1024


def test_claude_thinking_passthrough():
    request = MessagesRequest(
        model="antigravity/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Deep analysis"}],
        max_tokens=16384,
        thinking=ThinkingConfig(type="enabled", budget_tokens=8192),
    )
    cache: dict[str, str] = {}
    payload, backend_model = translate_messages_request(request, "proj-1", cache)

    assert backend_model == "claude-sonnet-4-6"
    gen_config = payload["request"]["generationConfig"]
    assert gen_config["thinkingConfig"]["thinkingBudget"] == 8192
    # For Claude, max_tokens is passed through natively without splitting
    assert gen_config["maxOutputTokens"] == 16384


def test_thinking_adaptive_type():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Explain relativity"}],
        max_tokens=8192,
        thinking={"type": "adaptive"},
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "proj-1", cache)

    gen_config = payload["request"]["generationConfig"]
    assert "thinkingConfig" in gen_config
    assert gen_config["thinkingConfig"]["thinkingBudget"] == 8192


def test_thinking_disabled_not_forwarded():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=2048,
        thinking=ThinkingConfig(type="disabled"),
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "proj-1", cache)

    gen_config = payload["request"]["generationConfig"]
    assert "thinkingConfig" not in gen_config
    assert gen_config["maxOutputTokens"] == 2048

