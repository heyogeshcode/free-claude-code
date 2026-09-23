"""Tests for Phase 1: System Prompt Fidelity in Antigravity provider."""

from free_claude_code.core.anthropic.models import MessagesRequest, SystemContent
from free_claude_code.providers.antigravity.translator import (
    _extract_system_text,
    extract_system_cache_control,
    is_known_harness_system_prompt,
    translate_messages_request,
)


def test_extract_system_text_string_exact_fidelity():
    complex_prompt = (
        "<system>\n"
        "  <instructions>\n"
        "    You are Claude Code, an expert coding assistant.\n"
        "    - Follow the rules in `CLAUDE.md`.\n"
        "    - Never guess filenames.\n"
        "  </instructions>\n"
        "  <context>\n"
        "    ```json\n"
        "    {\"repo\": \"free-claude-code\", \"branch\": \"main\"}\n"
        "    ```\n"
        "  </context>\n"
        "</system>"
    )
    result = _extract_system_text(complex_prompt)
    assert result == complex_prompt


def test_extract_system_text_multiblock_xml_structure():
    blocks = [
        {
            "type": "text",
            "text": "<instructions>\nBe precise and concise.\n</instructions>\n",
        },
        {
            "type": "text",
            "text": "<context>\nEnvironment: Linux x86_64\nPython: 3.14\n</context>",
        },
    ]
    result = _extract_system_text(blocks)
    expected = (
        "<instructions>\nBe precise and concise.\n</instructions>\n"
        "<context>\nEnvironment: Linux x86_64\nPython: 3.14\n</context>"
    )
    assert result == expected


def test_extract_system_text_preserves_cache_control_blocks():
    blocks = [
        SystemContent(
            type="text",
            text="Core instructions block.",
            cache_control={"type": "ephemeral"},
        ),
        SystemContent(
            type="text",
            text="Project CLAUDE.md context block.",
            cache_control={"type": "ephemeral"},
        ),
        SystemContent(
            type="text",
            text="Dynamic session context block.",
        ),
    ]
    result = _extract_system_text(blocks)
    assert "Core instructions block." in result
    assert "Project CLAUDE.md context block." in result
    assert "Dynamic session context block." in result

    cache_markers = extract_system_cache_control(blocks)
    assert len(cache_markers) == 2
    assert all(m.get("type") == "ephemeral" for m in cache_markers)


def test_extract_system_text_citations_and_references():
    blocks = [
        {"type": "text", "text": "Refer to the codebase guidelines [1]."},
        {"type": "reference", "text": "[1] Codebase Architecture Guide: docs/arch.md"},
    ]
    result = _extract_system_text(blocks)
    assert "Refer to the codebase guidelines [1]." in result
    assert "[1] Codebase Architecture Guide: docs/arch.md" in result


def test_harness_prompt_detection():
    claude_code_prompt = (
        "You are Claude Code, Anthropic's official CLI for software development.\n"
        "<claude_info>\nversion: 1.0\n</claude_info>"
    )
    antigravity_prompt = "<antigravity_instructions>\nBe helpful.\n</antigravity_instructions>"
    generic_prompt = "You are a poet who writes haikus."

    assert is_known_harness_system_prompt(claude_code_prompt) is True
    assert is_known_harness_system_prompt(antigravity_prompt) is True
    assert is_known_harness_system_prompt(generic_prompt) is False


def test_translate_messages_request_with_complex_system():
    complex_prompt = (
        "<system>\n"
        "  <instructions>Do not hallucinate</instructions>\n"
        "</system>"
    )
    request = MessagesRequest(
        model="antigravity/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hello"}],
        system=complex_prompt,
        max_tokens=1024,
    )
    cache: dict[str, str] = {}
    payload, _ = translate_messages_request(request, "test-proj", cache)

    sys_instruction = payload["request"]["systemInstruction"]
    assert sys_instruction == {"parts": [{"text": complex_prompt}]}

