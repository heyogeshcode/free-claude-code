"""Tests for Phase 3: Tool Calling Fidelity in Antigravity provider."""

from free_claude_code.core.anthropic.models import MessagesRequest, Tool
from free_claude_code.providers.antigravity.translator import (
    _find_tool_name_by_id,
    _sanitize_schema,
    translate_messages_request,
)


def test_sanitize_schema_const_mapping():
    schema = {
        "type": "object",
        "properties": {
            "action": {"const": "restart"},
            "count": {"const": 42},
            "enabled": {"const": True},
        },
    }
    sanitized = _sanitize_schema(schema)
    props = sanitized["properties"]
    assert props["action"]["enum"] == ["restart"]
    assert props["action"]["type"] == "STRING"
    assert props["count"]["enum"] == ["42"]
    assert props["count"]["type"] == "INTEGER"
    assert props["enabled"]["enum"] == ["True"]
    assert props["enabled"]["type"] == "BOOLEAN"


def test_sanitize_schema_defs_inlining():
    schema = {
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"zip": {"type": "string"}},
            }
        },
        "type": "object",
        "properties": {
            "home": {"$ref": "#/$defs/Address"},
        },
    }
    sanitized = _sanitize_schema(schema)
    home_prop = sanitized["properties"]["home"]
    assert home_prop["type"] == "OBJECT"
    assert "zip" in home_prop["properties"]
    assert home_prop["properties"]["zip"]["type"] == "STRING"


def test_sanitize_schema_missing_required_injected():
    schema = {
        "type": "object",
        "properties": {
            "existing_field": {"type": "string"},
        },
        "required": ["existing_field", "undeclared_required_arg"],
        "additionalProperties": True,
    }
    sanitized = _sanitize_schema(schema)
    assert "undeclared_required_arg" in sanitized["required"]
    assert "undeclared_required_arg" in sanitized["properties"]
    assert sanitized["properties"]["undeclared_required_arg"]["type"] == "STRING"
    assert sanitized["additionalProperties"] is True


def test_tool_choice_string_and_dict():
    # 1. String "any"
    req1 = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="any",
        tools=[Tool(name="t1", input_schema={"type": "object"})],
        max_tokens=100,
    )
    cache: dict[str, str] = {}
    payload1, _ = translate_messages_request(req1, "proj-1", cache)
    assert payload1["request"]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"

    # 2. String "auto"
    req2 = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="auto",
        tools=[Tool(name="t1", input_schema={"type": "object"})],
        max_tokens=100,
    )
    payload2, _ = translate_messages_request(req2, "proj-1", cache)
    assert payload2["request"]["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"

    # 3. Specific tool dict
    req3 = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice={"type": "tool", "name": "my_special_tool"},
        tools=[Tool(name="my_special_tool", input_schema={"type": "object"})],
        max_tokens=100,
    )
    payload3, _ = translate_messages_request(req3, "proj-1", cache)
    fn_cfg = payload3["request"]["toolConfig"]["functionCallingConfig"]
    assert fn_cfg["mode"] == "ANY"
    assert fn_cfg["allowedFunctionNames"] == ["my_special_tool"]


def test_unmatched_tool_result_recovers_recent_tool_call():
    messages = [
        {"role": "user", "content": "Please inspect files"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_abc",
                    "name": "read_dir",
                    "input": {"path": "/tmp"},
                }
            ],
        },
    ]
    # Tool result has no tool_id or unmatched id
    recovered_name = _find_tool_name_by_id(messages, "unmatched_id_xyz")
    assert recovered_name == "read_dir"

    recovered_empty = _find_tool_name_by_id(messages, "")
    assert recovered_empty == "read_dir"

