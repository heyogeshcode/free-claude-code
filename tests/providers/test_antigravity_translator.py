"""Tests for Antigravity request/response translation and stream processing."""

from free_claude_code.core.anthropic.models import (
    MessagesRequest,
    Tool,
)
from free_claude_code.providers.antigravity.translator import (
    AntigravityStreamTranslator,
    translate_messages_request,
)
from tests.providers.request_factory import make_messages_request


def test_translate_simple_text_request():
    request = make_messages_request(
        "antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "Hello Antigravity"}],
        system="You are a helpful assistant.",
        max_tokens=1024,
    )
    cache: dict[str, str] = {}
    payload, backend_model = translate_messages_request(request, "test-project", cache)

    assert backend_model == "gemini-3.8-flash-tiered"
    assert payload["project"] == "test-project"
    assert payload["model"] == "gemini-3.8-flash-tiered"

    inner = payload["request"]
    assert inner["systemInstruction"] == {
        "parts": [{"text": "You are a helpful assistant."}]
    }
    assert inner["generationConfig"]["maxOutputTokens"] == 1024
    assert len(inner["contents"]) == 1
    assert inner["contents"][0]["role"] == "user"
    assert inner["contents"][0]["parts"][0]["text"] == "Hello Antigravity"


def test_translate_tools_and_tool_results_with_signature():
    tool = Tool(
        name="weather",
        description="Get weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )

    request = MessagesRequest(
        model="antigravity/claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "weather",
                        "input": {"city": "Tokyo"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Sunny, 22C",
                    }
                ],
            },
        ],
        tools=[tool],
        max_tokens=2048,
    )

    cache: dict[str, str] = {"call_123": "sig_abc123"}
    payload, backend_model = translate_messages_request(request, "test-project", cache)

    assert backend_model == "claude-sonnet-4-6"
    inner = payload["request"]
    assert len(inner["tools"]) == 1
    fn_decl = inner["tools"][0]["functionDeclarations"][0]
    assert fn_decl["name"] == "weather"
    assert fn_decl["parameters"]["type"] == "OBJECT"

    contents = inner["contents"]
    assert len(contents) == 3

    # First turn: user
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "What's the weather in Tokyo?"

    # Second turn: model tool_use with thoughtSignature replayed from cache
    assert contents[1]["role"] == "model"
    part = contents[1]["parts"][0]
    assert "functionCall" in part
    assert part["functionCall"]["name"] == "weather"
    assert part["functionCall"]["args"] == {"city": "Tokyo"}
    assert part["thoughtSignature"] == "sig_abc123"

    # Third turn: user functionResponse
    assert contents[2]["role"] == "user"
    resp_part = contents[2]["parts"][0]
    assert "functionResponse" in resp_part
    assert resp_part["functionResponse"]["name"] == "weather"
    assert resp_part["functionResponse"]["response"]["content"] == "Sunny, 22C"


def test_stream_translator_text_events():
    cache: dict[str, str] = {}
    translator = AntigravityStreamTranslator(
        "antigravity/gemini-3.8-flash", cache, input_tokens=15
    )

    line1 = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}], "usageMetadata": {"candidatesTokenCount": 2}}}'
    line2 = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "world!"}], "finishReason": "STOP"}}]}}'

    events1 = translator.process_line(line1)
    events2 = translator.process_line(line2)
    final_events = translator.finalize()

    all_events = events1 + events2 + final_events
    raw_joined = "".join(all_events)

    assert "event: message_start" in raw_joined
    assert "antigravity/gemini-3.8-flash" in raw_joined
    assert "event: content_block_start" in raw_joined
    assert "event: content_block_delta" in raw_joined
    assert "Hello " in raw_joined
    assert "world!" in raw_joined
    assert "event: content_block_stop" in raw_joined
    assert "event: message_delta" in raw_joined
    assert '"stop_reason": "end_turn"' in raw_joined
    assert "event: message_stop" in raw_joined


def test_stream_translator_captures_thought_signature():
    cache: dict[str, str] = {}
    translator = AntigravityStreamTranslator("antigravity/gemini-3.8-flash", cache)

    line = (
        'data: {"response": {"candidates": [{"content": {"parts": [{'
        '"thoughtSignature": "test_thought_sig_xyz",'
        '"functionCall": {"name": "run_bash", "args": {"cmd": "ls"}, "id": "call_456"}'
        "}]}}]}}"
    )

    events = translator.process_line(line)
    final_events = translator.finalize()
    all_events = events + final_events
    raw_joined = "".join(all_events)

    # Verify signature was recorded in cache
    assert cache.get("call_456") == "test_thought_sig_xyz"
    assert cache.get("run_bash") == "test_thought_sig_xyz"
    assert cache.get("latest_turn_sig") == "test_thought_sig_xyz"

    # Verify tool_use events
    assert "tool_use" in raw_joined
    assert "run_bash" in raw_joined
    assert '"stop_reason": "tool_use"' in raw_joined
