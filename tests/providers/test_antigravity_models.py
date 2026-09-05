"""Tests for Antigravity live model discovery and alias resolution."""

from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.antigravity.models import (
    DEFAULT_ANTIGRAVITY_MODELS,
    parse_antigravity_models,
    resolve_backend_model,
)


def test_default_models_include_gemini_and_claude():
    model_ids = {m.model_id for m in DEFAULT_ANTIGRAVITY_MODELS}
    assert "gemini-3.8-flash" in model_ids
    assert "gemini-3.8-flash-tiered" in model_ids
    assert "gemini-3.7-flash" in model_ids
    assert "claude-sonnet-4-6" in model_ids
    assert "claude-opus-4-6-thinking" in model_ids

    gemini_38 = next(
        m for m in DEFAULT_ANTIGRAVITY_MODELS if m.model_id == "gemini-3.8-flash"
    )
    assert gemini_38.supports_thinking
    assert ModelInputModality.TEXT in gemini_38.input_modalities
    assert ModelInputModality.IMAGE in gemini_38.input_modalities

    claude_sonnet = next(
        m for m in DEFAULT_ANTIGRAVITY_MODELS if m.model_id == "claude-sonnet-4-6"
    )
    assert claude_sonnet.supports_thinking
    assert ModelInputModality.TEXT in claude_sonnet.input_modalities


def test_resolve_backend_model():
    # Direct with prefix
    assert (
        resolve_backend_model("antigravity/gemini-3.8-flash")
        == "gemini-3.8-flash-tiered"
    )
    assert resolve_backend_model("antigravity/claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert (
        resolve_backend_model("antigravity/claude-opus-4-6-thinking")
        == "claude-opus-4-6-thinking"
    )

    # Direct without prefix
    assert resolve_backend_model("gemini-3.8-flash") == "gemini-3.8-flash-tiered"
    assert resolve_backend_model("gemini-3.7-flash") == "gemini-3.7-flash-tiered"
    assert resolve_backend_model("claude-sonnet") == "claude-sonnet-4-6"
    assert resolve_backend_model("claude-opus") == "claude-opus-4-6-thinking"

    # Exact backend ID passed directly
    assert resolve_backend_model("gemini-3.8-flash-tiered") == "gemini-3.8-flash-tiered"
    assert (
        resolve_backend_model("antigravity/gemini-3.8-flash-tiered")
        == "gemini-3.8-flash-tiered"
    )


def test_parse_antigravity_models_empty_or_malformed():
    models = parse_antigravity_models({})
    assert models == frozenset(DEFAULT_ANTIGRAVITY_MODELS)

    models_none = parse_antigravity_models({"models": None})
    assert models_none == frozenset(DEFAULT_ANTIGRAVITY_MODELS)


def test_parse_antigravity_models_from_live_response():
    payload = {
        "models": {
            "gemini-3.8-flash-tiered": {
                "displayName": "Gemini 3.8 Flash (Tiered)",
                "supportsThinking": True,
                "supportsImages": True,
                "maxTokens": 1048576,
                "maxOutputTokens": 65536,
            },
            "claude-sonnet-4-6": {
                "displayName": "Claude 3.7 Sonnet (v4.6)",
                "supportsThinking": True,
                "supportsImages": True,
                "maxTokens": 200000,
                "maxOutputTokens": 65536,
            },
            "chat_internal_debug": {
                "displayName": "Debug",
            },
        }
    }

    parsed = parse_antigravity_models(payload)
    parsed_ids = {m.model_id for m in parsed}

    # Should contain exact model IDs
    assert "gemini-3.8-flash-tiered" in parsed_ids
    assert "claude-sonnet-4-6" in parsed_ids

    # Should automatically create clean alias for -tiered
    assert "gemini-3.8-flash" in parsed_ids

    # Should filter internal debug IDs
    assert "chat_internal_debug" not in parsed_ids

    # Check capabilities
    gemini_clean = next(m for m in parsed if m.model_id == "gemini-3.8-flash")
    assert gemini_clean.supports_thinking
    assert ModelInputModality.IMAGE in gemini_clean.input_modalities
    assert gemini_clean.context_window_tokens == 1048576
