"""Tests for Phase 4: Conversation Structure Integrity in Antigravity provider."""

from free_claude_code.core.anthropic.models import MessagesRequest, ThinkingConfig
from free_claude_code.providers.antigravity.translator import translate_messages_request


def test_replayed_thinking_includes_thought_signature():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user is asking a basic arithmetic question...",
                        "signature": "sig_thought_999",
                    },
                    {"type": "text", "text": "2 + 2 = 4."},
                ],
            },
            {"role": "user", "content": "And 4+4?"},
        ],
        max_tokens=2048,
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "test-proj", cache)

    contents = payload["request"]["contents"]
    assert len(contents) == 3
    # Turn 1: model
    model_turn = contents[1]
    assert model_turn["role"] == "model"
    thought_part = model_turn["parts"][0]
    assert thought_part["thought"] is True
    assert thought_part["text"] == "The user is asking a basic arithmetic question..."
    assert thought_part["thoughtSignature"] == "sig_thought_999"

    text_part = model_turn["parts"][1]
    assert text_part["text"] == "2 + 2 = 4."


def test_gemini_turn_ordering_no_synthetic_words():
    # Assistant-only message
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[
            {"role": "assistant", "content": "System ready."},
        ],
        max_tokens=1000,
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "test-proj", cache)

    contents = payload["request"]["contents"]
    # Check that first and last turns are user, but NO "Hello" or "Please continue."
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == " "
    assert "hello" not in contents[0]["parts"][0]["text"].lower()

    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "System ready."

    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["text"] == " "
    assert "please continue" not in contents[2]["parts"][0]["text"].lower()


def test_future_gemini_4_pro_model_compatibility():
    """Verify future models like gemini-4-pro automatically get optimal handling."""
    request = MessagesRequest(
        model="antigravity/gemini-4-pro",
        messages=[{"role": "user", "content": "Hello future Gemini 4"}],
        max_tokens=16384,
        thinking=ThinkingConfig(type="enabled", budget_tokens=8192),
    )
    cache: dict[str, str] = {}
    payload, backend_model = translate_messages_request(request, "test-proj", cache)

    assert backend_model == "gemini-4-pro"
    gen_config = payload["request"]["generationConfig"]
    # Verify generic Gemini defaults are automatically provided for unknown Gemini models
    assert gen_config["temperature"] == 1.0
    assert gen_config["topP"] == 0.95
    assert gen_config["topK"] == 64
    # Verify thinking budget splitting works for Gemini 4 Pro
    assert gen_config["thinkingConfig"]["thinkingBudget"] == 8192
    assert gen_config["maxOutputTokens"] == 8192  # 16384 - 8192 = 8192


def test_empty_messages_fallback():
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[],
        max_tokens=1000,
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "test-proj", cache)

    contents = payload["request"]["contents"]
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == " "

