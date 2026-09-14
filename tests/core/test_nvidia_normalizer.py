"""Unit tests for NvidiaSchemaNormalizer."""

import json

from free_claude_code.core.nvidia_normalizer import NvidiaSchemaNormalizer


def test_anthropic_to_openai_chat_system_and_messages():
    anthropic_req = {
        "model": "claude-3-5-sonnet",
        "system": "You are a helpful coding assistant.",
        "messages": [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    openai_body = NvidiaSchemaNormalizer.anthropic_to_openai_chat(anthropic_req)

    assert openai_body["model"] == "meta/llama-3.3-70b-instruct"
    assert len(openai_body["messages"]) == 3
    assert openai_body["messages"][0] == {
        "role": "system",
        "content": "You are a helpful coding assistant.",
    }
    assert openai_body["messages"][1] == {"role": "user", "content": "Hello world"}
    assert openai_body["messages"][2] == {"role": "assistant", "content": "Hi there!"}
    assert openai_body["max_tokens"] == 1024
    assert openai_body["temperature"] == 0.7
    assert openai_body["stream"] is True


def test_anthropic_to_openai_chat_tools_and_tool_results():
    anthropic_req = {
        "model": "nvidia_fallback/meta/llama-3.3-70b-instruct",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "What is the weather in SF?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "San Francisco"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "65 degrees and sunny",
                    }
                ],
            },
        ],
    }
    openai_body = NvidiaSchemaNormalizer.anthropic_to_openai_chat(anthropic_req)

    # Provider prefix stripped
    assert openai_body["model"] == "meta/llama-3.3-70b-instruct"
    assert len(openai_body["tools"]) == 1
    assert openai_body["tools"][0]["function"]["name"] == "get_weather"

    msgs = openai_body["messages"]
    # User message
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is the weather in SF?"

    # Assistant message with tool_calls
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Let me check..."
    assert len(msgs[1]["tool_calls"]) == 1
    assert msgs[1]["tool_calls"][0]["id"] == "toolu_123"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {
        "city": "San Francisco"
    }

    # Tool result message
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "toolu_123"
    assert msgs[2]["content"] == "65 degrees and sunny"


def test_openai_chunk_to_anthropic_events_text_and_finish():
    state = {}
    model = "meta/llama-3.3-70b-instruct"
    msg_id = "msg_test_01"

    # 1. First chunk with text
    chunk1 = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
    events1 = NvidiaSchemaNormalizer.openai_chunk_to_anthropic_events(
        chunk1, state, model=model, message_id=msg_id
    )

    # Should contain message_start, content_block_start, content_block_delta
    assert any("event: message_start" in ev for ev in events1)
    assert any("event: content_block_start" in ev for ev in events1)
    assert any("event: content_block_delta" in ev for ev in events1)
    assert any('"text": "Hello"' in ev for ev in events1)

    # 2. Second chunk with text
    chunk2 = {"choices": [{"delta": {"content": " world!"}, "finish_reason": None}]}
    events2 = NvidiaSchemaNormalizer.openai_chunk_to_anthropic_events(
        chunk2, state, model=model, message_id=msg_id
    )
    assert len(events2) == 1
    assert "event: content_block_delta" in events2[0]
    assert '"text": " world!"' in events2[0]

    # 3. Final chunk with finish_reason
    chunk3 = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    events3 = NvidiaSchemaNormalizer.openai_chunk_to_anthropic_events(
        chunk3, state, model=model, message_id=msg_id
    )
    assert any("event: content_block_stop" in ev for ev in events3)
    assert any("event: message_delta" in ev for ev in events3)
    assert any("event: message_stop" in ev for ev in events3)


def test_openai_chunk_to_anthropic_events_reasoning():
    state = {}
    model = "deepseek-ai/deepseek-r1"
    msg_id = "msg_test_r1"

    # Chunk with reasoning content
    chunk_r = {
        "choices": [
            {
                "delta": {"reasoning_content": "Let me think deeply"},
                "finish_reason": None,
            }
        ]
    }
    events = NvidiaSchemaNormalizer.openai_chunk_to_anthropic_events(
        chunk_r, state, model=model, message_id=msg_id
    )
    assert any("event: message_start" in ev for ev in events)
    assert any("event: content_block_start" in ev for ev in events)
    assert any('"type": "thinking"' in ev for ev in events)
    assert any('"thinking": "Let me think deeply"' in ev for ev in events)
