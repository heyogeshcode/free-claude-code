"""Tests for Phases 5 to 16 optimizations in Antigravity provider and translator."""

import json
import pytest
from unittest.mock import MagicMock

from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
    Tool,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.providers.antigravity.models import (
    DEFAULT_ANTIGRAVITY_MODELS,
    MODEL_GENERATION_DEFAULTS,
)
from free_claude_code.providers.antigravity.provider import _sanitize_responses_request
from free_claude_code.providers.antigravity.translator import (
    AntigravityStreamTranslator,
    translate_messages_request,
)


def test_phase_5_response_format_json_object():
    """Phase 5: response_format with json_object sets responseMimeType to application/json."""
    req = MessagesRequest(
        model="gemini-3.8-flash",
        max_tokens=4096,
        messages=[Message(role="user", content="Return JSON")],
        response_format={"type": "json_object"},
    )
    cache = {}
    payload, _ = translate_messages_request(req, "test-proj", cache)
    gen_config = payload["request"]["generationConfig"]
    assert gen_config.get("responseMimeType") == "application/json"


def test_phase_5_response_format_json_schema():
    """Phase 5: response_format with json_schema sets responseSchema."""
    req = MessagesRequest(
        model="gemini-3.8-flash",
        max_tokens=4096,
        messages=[Message(role="user", content="Return structured JSON")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        },
    )
    cache = {}
    payload, _ = translate_messages_request(req, "test-proj", cache)
    gen_config = payload["request"]["generationConfig"]
    assert gen_config.get("responseMimeType") == "application/json"
    assert "responseSchema" in gen_config
    assert gen_config["responseSchema"]["type"] == "OBJECT"
    assert "name" in gen_config["responseSchema"]["properties"]


def test_phase_13_pdf_document_modality():
    """Phase 13: Document modality (PDF) is preserved as inlineData."""
    req = MessagesRequest(
        model="gemini-3.8-flash",
        max_tokens=4096,
        messages=[
            Message(
                role="user",
                content=[
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQK...",
                        },
                    },
                    {"type": "text", "text": "Summarize this PDF"},
                ],
            )
        ],
    )
    cache = {}
    payload, _ = translate_messages_request(req, "test-proj", cache)
    parts = payload["request"]["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[0]["inlineData"]["mimeType"] == "application/pdf"
    assert parts[0]["inlineData"]["data"] == "JVBERi0xLjQK..."
    assert parts[1]["text"] == "Summarize this PDF"


def test_phase_13_multimodal_tool_result_with_image_and_error():
    """Phase 13: Multimodal tool results preserve images as inlineData and signal error."""
    req = MessagesRequest(
        model="gemini-3.8-flash",
        max_tokens=4096,
        messages=[
            Message(
                role="assistant",
                content=[
                    {
                        "type": "tool_use",
                        "id": "call_screenshot_1",
                        "name": "take_screenshot",
                        "input": {},
                    }
                ],
            ),
            Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_screenshot_1",
                        "is_error": True,
                        "content": [
                            {"type": "text", "text": "Failed partially"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgoAAA...",
                                },
                            },
                        ],
                    }
                ],
            ),
        ],
    )
    cache = {}
    payload, _ = translate_messages_request(req, "test-proj", cache)
    contents = payload["request"]["contents"]
    user_turn = contents[-1]
    assert user_turn["role"] == "user"
    parts = user_turn["parts"]
    assert len(parts) == 2
    # First part is functionResponse
    fn_resp = parts[0]["functionResponse"]
    assert fn_resp["name"] == "take_screenshot"
    assert fn_resp["response"]["error"] is True
    assert "Error: Failed partially" in fn_resp["response"]["content"]
    # Second part is inlineData image
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert parts[1]["inlineData"]["data"] == "iVBORw0KGgoAAA..."


def test_phase_14_and_15_token_and_cache_tracking():
    """Phase 14 & 15: Stream translator captures promptTokenCount and cachedContentTokenCount."""
    cache = {}
    translator = AntigravityStreamTranslator("gemini-3.8-flash", cache, input_tokens=100)

    # First chunk with usageMetadata
    chunk = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello world"}]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 150,
                "candidatesTokenCount": 2,
                "cachedContentTokenCount": 50,
            },
        }
    }
    events = translator.process_line(f"data: {json.dumps(chunk)}")
    assert len(events) >= 2  # message_start, content_block_start, content_block_delta

    # Verify message_start contains updated promptTokenCount and cache_read_input_tokens
    msg_start_ev = json.loads(events[0].partition("data: ")[2])
    assert msg_start_ev["type"] == "message_start"
    assert msg_start_ev["message"]["usage"]["input_tokens"] == 150
    assert msg_start_ev["message"]["usage"]["cache_read_input_tokens"] == 50

    # Finalize should also emit cache_read_input_tokens
    final_events = translator.finalize()
    delta_ev = [json.loads(e.partition("data: ")[2]) for e in final_events if "message_delta" in e][0]
    assert delta_ev["usage"]["cache_read_input_tokens"] == 50
    assert delta_ev["usage"]["output_tokens"] == 2


def test_phase_16_stream_buffer_oom_protection():
    """Phase 16: An excessively long buffer without valid JSON raises ValueError to prevent OOM."""
    cache = {}
    translator = AntigravityStreamTranslator("gemini-3.8-flash", cache)

    # Feed data larger than 5MB
    large_garbage = "x" * (5 * 1024 * 1024 + 100)
    with pytest.raises(ValueError, match="exceeded 5MB limit"):
        translator.process_line(f"data: {large_garbage}")


def test_phase_8_sanitize_responses_request():
    """Phase 8: _sanitize_responses_request removes unsupported truncation and cleans reasoning."""
    req = OpenAIResponsesRequest(
        model="gemini-3.8-flash",
        input=[{"type": "message", "role": "user", "content": "Hi"}],
        truncation="auto",
        reasoning={"effort": "high", "summary": "none"},
    )
    sanitized = _sanitize_responses_request(req)
    dumped = sanitized.model_dump(exclude_none=True)
    assert "truncation" not in dumped  # "auto" removed so Messages converter won't fail
    assert dumped["reasoning"]["effort"] == "high"
    assert "summary" not in dumped["reasoning"]  # "none" removed, only "auto" preserved


def test_phase_10_model_capabilities_and_defaults():
    """Phase 10: Verify modern max output tokens and model defaults."""
    # Check default model info max_output_tokens
    gemini_models = [m for m in DEFAULT_ANTIGRAVITY_MODELS if m.model_id.startswith("gemini")]
    claude_models = [m for m in DEFAULT_ANTIGRAVITY_MODELS if m.model_id.startswith("claude") or m.model_id.startswith("sonnet")]
    
    for m in gemini_models:
        assert m.max_output_tokens == 65_536
    for m in claude_models:
        assert m.max_output_tokens == 64_000

    # Check generation defaults
    assert "claude-sonnet-4-6" in MODEL_GENERATION_DEFAULTS
    assert MODEL_GENERATION_DEFAULTS["claude-sonnet-4-6"]["temperature"] == 1.0
