"""Translation between Anthropic Messages protocol and Antigravity streamGenerateContent API."""

import json
import uuid
from typing import Any

from loguru import logger

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming.emitter import format_sse_event

from .models import resolve_backend_model


def translate_messages_request(
    request: MessagesRequest,
    project_id: str,
    signature_cache: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """Translate Anthropic MessagesRequest into Antigravity streamGenerateContent payload."""
    backend_model = resolve_backend_model(request.model)

    req_payload: dict[str, Any] = {}

    # System instruction
    system_text = _extract_system_text(request.system)
    if system_text:
        req_payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    # Tools
    if request.tools:
        function_declarations: list[dict[str, Any]] = []
        for tool in request.tools:
            name = getattr(tool, "name", None) or (
                tool.get("name") if isinstance(tool, dict) else ""
            )
            description = getattr(tool, "description", None) or (
                tool.get("description") if isinstance(tool, dict) else ""
            )
            schema = getattr(tool, "input_schema", None) or (
                tool.get("input_schema") if isinstance(tool, dict) else {}
            )
            function_declarations.append(
                {
                    "name": name,
                    "description": description or "",
                    "parameters": _sanitize_schema(schema),
                }
            )
        if function_declarations:
            req_payload["tools"] = [{"functionDeclarations": function_declarations}]

    # Contents (alternating user / model messages)
    contents = _build_contents(request.messages, signature_cache)
    if not contents:
        # Fallback empty turn
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    req_payload["contents"] = contents

    # Generation config
    gen_config: dict[str, Any] = {}
    if request.max_tokens:
        gen_config["maxOutputTokens"] = request.max_tokens
    if request.temperature is not None:
        gen_config["temperature"] = request.temperature

    thinking = getattr(request, "thinking", None)
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int) and budget > 0:
            gen_config["thinkingConfig"] = {"thinkingBudget": budget}

    if gen_config:
        req_payload["generationConfig"] = gen_config

    full_payload = {
        "project": project_id,
        "model": backend_model,
        "request": req_payload,
    }
    return full_payload, backend_model


def _extract_system_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "\n\n".join(parts)
    return ""


def _sanitize_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}

    result = dict(schema)
    # Strip keywords unsupported by Google Gemini schema validator
    for forbidden in ("$schema", "additionalProperties", "title", "$defs", "definitions"):
        result.pop(forbidden, None)

    # Ensure type is uppercase or valid JSON schema
    schema_type = result.get("type")
    if isinstance(schema_type, str):
        result["type"] = (
            schema_type.upper()
            if schema_type.lower()
            in ("object", "string", "number", "integer", "boolean", "array")
            else schema_type
        )

    if "properties" in result and isinstance(result["properties"], dict):
        sanitized_props: dict[str, Any] = {}
        for prop_name, prop_def in result["properties"].items():
            sanitized_props[prop_name] = _sanitize_schema(prop_def)
        result["properties"] = sanitized_props

    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _sanitize_schema(result["items"])

    return result


def _build_contents(
    messages: list[Any],
    signature_cache: dict[str, str],
) -> list[dict[str, Any]]:
    raw_turns: list[dict[str, Any]] = []

    # First pass: look for any thinking signatures in assistant messages
    for msg in messages:
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, list):
            for block in content:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else ""
                )
                sig = getattr(block, "signature", None) or (
                    block.get("signature") if isinstance(block, dict) else ""
                )
                if btype == "thinking" and sig:
                    signature_cache["latest_turn_sig"] = sig

    for msg in messages:
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else "user"
        )
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else ""
        )
        gemini_role = "model" if role == "assistant" else "user"

        parts: list[dict[str, Any]] = []

        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else ""
                )
                if btype == "text":
                    text_val = getattr(block, "text", None) or (
                        block.get("text") if isinstance(block, dict) else ""
                    )
                    if text_val:
                        parts.append({"text": text_val})
                elif btype == "image":
                    source = getattr(block, "source", None) or (
                        block.get("source") if isinstance(block, dict) else {}
                    )
                    data_val = (
                        source.get("data")
                        if isinstance(source, dict)
                        else getattr(source, "data", None)
                    )
                    media_type = (
                        source.get("media_type")
                        if isinstance(source, dict)
                        else getattr(source, "media_type", "image/jpeg")
                    )
                    if data_val:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": media_type or "image/jpeg",
                                    "data": data_val,
                                }
                            }
                        )
                elif btype == "tool_use":
                    tool_id = getattr(block, "id", None) or (
                        block.get("id") if isinstance(block, dict) else ""
                    )
                    tool_name = getattr(block, "name", None) or (
                        block.get("name") if isinstance(block, dict) else ""
                    )
                    tool_input = getattr(block, "input", None) or (
                        block.get("input") if isinstance(block, dict) else {}
                    )
                    if not isinstance(tool_input, dict):
                        tool_input = {}

                    part: dict[str, Any] = {
                        "functionCall": {
                            "name": tool_name,
                            "args": tool_input,
                            "id": tool_id,
                        }
                    }
                    # Retrieve cached thought signature if available
                    sig = (
                        signature_cache.get(tool_id)
                        or signature_cache.get(tool_name)
                        or signature_cache.get("latest_turn_sig")
                    )
                    if sig:
                        part["thoughtSignature"] = sig
                    parts.append(part)
                elif btype == "tool_result":
                    tool_use_id = getattr(block, "tool_use_id", None) or (
                        block.get("tool_use_id") if isinstance(block, dict) else ""
                    )
                    result_content = getattr(block, "content", None) or (
                        block.get("content") if isinstance(block, dict) else ""
                    )
                    if isinstance(result_content, list):
                        # Combine text parts
                        text_items = [
                            c.get("text", "")
                            if isinstance(c, dict)
                            else getattr(c, "text", "")
                            for c in result_content
                        ]
                        result_str = "\n".join(filter(None, text_items))
                    else:
                        result_str = str(result_content)

                    # Wrap in functionResponse
                    part = {
                        "functionResponse": {
                            "name": _find_tool_name_by_id(
                                messages, tool_use_id, signature_cache
                            )
                            or "tool",
                            "response": {"content": result_str},
                            "id": tool_use_id,
                        }
                    }
                    parts.append(part)

        if parts:
            raw_turns.append({"role": gemini_role, "parts": parts})

    # Merge consecutive turns with identical role into one turn
    merged: list[dict[str, Any]] = []
    for turn in raw_turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["parts"].extend(turn["parts"])
        else:
            merged.append(turn)

    return merged


def _find_tool_name_by_id(
    messages: list[Any],
    tool_id: str,
    signature_cache: dict[str, str] | None = None,
) -> str | None:
    for msg in messages:
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, list):
            for block in content:
                bid = getattr(block, "id", None) or (
                    block.get("id") if isinstance(block, dict) else ""
                )
                bname = getattr(block, "name", None) or (
                    block.get("name") if isinstance(block, dict) else ""
                )
                if bid == tool_id and bname:
                    return bname
    if signature_cache:
        cached_name = signature_cache.get(f"name_{tool_id}")
        if cached_name:
            return cached_name
    return None


class AntigravityStreamTranslator:
    """Translates SSE streaming lines from streamGenerateContent to Anthropic SSE events."""

    def __init__(
        self,
        public_model: str,
        signature_cache: dict[str, str],
        *,
        input_tokens: int = 0,
    ) -> None:
        self.public_model = public_model
        self.signature_cache = signature_cache
        self.input_tokens = input_tokens
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._message_started = False
        self._current_block_index = -1
        self._current_block_type: str | None = None
        self._total_output_tokens = 0
        self._stop_reason = "end_turn"

    def process_line(self, line: str) -> list[str]:
        """Process one SSE line and return zero or more Anthropic SSE event frames."""
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return []

        payload_str = stripped[5:].strip()
        if not payload_str:
            return []

        try:
            chunk = json.loads(payload_str)
        except json.JSONDecodeError:
            logger.debug("Failed to decode SSE JSON: {}", payload_str[:100])
            return []

        events: list[str] = []

        # Start message on first valid payload
        if not self._message_started:
            self._message_started = True
            events.append(
                format_sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": self._message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": self.public_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": self.input_tokens,
                                "output_tokens": 1,
                            },
                        },
                    },
                )
            )

        resp = chunk.get("response", {})
        candidates = resp.get("candidates", [])
        usage_meta = resp.get("usageMetadata", {})
        if usage_meta:
            candidates_tokens = usage_meta.get("candidatesTokenCount")
            if isinstance(candidates_tokens, int):
                self._total_output_tokens = candidates_tokens

        for candidate in candidates:
            finish_reason = candidate.get("finishReason")
            if finish_reason:
                if finish_reason == "MAX_TOKENS":
                    self._stop_reason = "max_tokens"
                elif finish_reason == "STOP" and self._stop_reason != "tool_use":
                    self._stop_reason = "end_turn"

            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                # Capture and cache thoughtSignature if present
                sig = part.get("thoughtSignature")
                if sig and isinstance(sig, str):
                    self.signature_cache["latest_turn_sig"] = sig

                # Text delta
                text = part.get("text")
                if text is not None and len(text) > 0:
                    if self._current_block_type != "text":
                        # Close previous block if open
                        if self._current_block_type is not None:
                            events.append(
                                format_sse_event(
                                    "content_block_stop",
                                    {
                                        "type": "content_block_stop",
                                        "index": self._current_block_index,
                                    },
                                )
                            )
                        self._current_block_index += 1
                        self._current_block_type = "text"
                        events.append(
                            format_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": self._current_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                        )
                    events.append(
                        format_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self._current_block_index,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )
                    )

                # Tool use / functionCall
                function_call = part.get("functionCall")
                if function_call and isinstance(function_call, dict):
                    self._stop_reason = "tool_use"
                    fn_name = function_call.get("name", "")
                    fn_args = function_call.get("args", {})
                    call_id = (
                        function_call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}"
                    )

                    # If this part also has thoughtSignature, record it for this tool call ID
                    self.signature_cache[f"name_{call_id}"] = fn_name
                    if sig:
                        self.signature_cache[call_id] = sig
                        self.signature_cache[fn_name] = sig

                    if self._current_block_type is not None:
                        events.append(
                            format_sse_event(
                                "content_block_stop",
                                {
                                    "type": "content_block_stop",
                                    "index": self._current_block_index,
                                },
                            )
                        )
                        self._current_block_type = None

                    self._current_block_index += 1
                    events.append(
                        format_sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": self._current_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call_id,
                                    "name": fn_name,
                                    "input": {},
                                },
                            },
                        )
                    )
                    events.append(
                        format_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self._current_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(fn_args),
                                },
                            },
                        )
                    )
                    events.append(
                        format_sse_event(
                            "content_block_stop",
                            {
                                "type": "content_block_stop",
                                "index": self._current_block_index,
                            },
                        )
                    )

        return events

    def finalize(self) -> list[str]:
        """Finalize the stream and emit terminal Anthropic SSE events."""
        events: list[str] = []
        if self._current_block_type is not None:
            events.append(
                format_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._current_block_index},
                )
            )
            self._current_block_type = None

        if not self._message_started:
            # Emit empty message if stream had no content
            events.append(
                format_sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": self._message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": self.public_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": self.input_tokens,
                                "output_tokens": 1,
                            },
                        },
                    },
                )
            )

        events.append(
            format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": max(1, self._total_output_tokens),
                    },
                },
            )
        )
        events.append(format_sse_event("message_stop", {"type": "message_stop"}))
        return events
