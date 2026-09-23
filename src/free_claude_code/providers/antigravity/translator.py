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
    contents = _build_contents(
        request.messages, signature_cache, backend_model=backend_model
    )
    if not contents:
        # Fallback empty turn
        contents = [{"role": "user", "parts": [{"text": " "}]}]
    req_payload["contents"] = contents

    # Generation config
    gen_config: dict[str, Any] = {}
    if request.temperature is not None:
        gen_config["temperature"] = request.temperature
    if request.stop_sequences:
        gen_config["stopSequences"] = list(request.stop_sequences)
    if request.top_p is not None:
        gen_config["topP"] = request.top_p
    if request.top_k is not None:
        gen_config["topK"] = request.top_k

    # Resolve thinking and budget from both Pydantic ThinkingConfig and dict representations
    thinking = request.thinking
    thinking_enabled = False
    budget: int | None = None

    if thinking is not None:
        if isinstance(thinking, dict):
            ttype = thinking.get("type")
            tenabled = thinking.get("enabled")
            tbudget = thinking.get("budget_tokens")
        else:
            ttype = getattr(thinking, "type", None)
            tenabled = getattr(thinking, "enabled", None)
            tbudget = getattr(thinking, "budget_tokens", None)

        if isinstance(tbudget, int) and not isinstance(tbudget, bool) and tbudget > 0:
            budget = tbudget

        if ttype == "disabled" or tenabled is False:
            thinking_enabled = False
        elif (
            ttype in ("enabled", "adaptive")
            or tenabled is True
            or (budget is not None)
        ):
            thinking_enabled = True

    if thinking_enabled:
        default_budget = min(32_768, request.max_tokens) if request.max_tokens else 32_768
        effective_budget = budget if budget is not None else default_budget
        gen_config["thinkingConfig"] = {"thinkingBudget": effective_budget}

        if backend_model.startswith("gemini"):
            # Gemini budget splitting: maxOutputTokens represents response output capacity
            if request.max_tokens:
                if request.max_tokens > effective_budget:
                    gen_config["maxOutputTokens"] = max(1024, request.max_tokens - effective_budget)
                else:
                    gen_config["maxOutputTokens"] = max(1024, request.max_tokens)
        else:
            # Claude models on Antigravity: forward max_tokens natively
            if request.max_tokens:
                gen_config["maxOutputTokens"] = request.max_tokens
    elif request.max_tokens:
        gen_config["maxOutputTokens"] = request.max_tokens

    # Response format / JSON mode
    rf = getattr(request, "response_format", None)
    if not rf and hasattr(request, "extra_body") and isinstance(request.extra_body, dict):
        rf = request.extra_body.get("response_format")
    if not rf and hasattr(request, "output_config") and isinstance(request.output_config, dict):
        rf = request.output_config.get("format")

    if rf:
        rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
        if rf_type == "json_object":
            gen_config["responseMimeType"] = "application/json"
        elif rf_type == "json_schema":
            gen_config["responseMimeType"] = "application/json"
            schema = None
            if isinstance(rf, dict):
                js = rf.get("json_schema", {})
                schema = js.get("schema", {}) if isinstance(js, dict) else getattr(js, "schema", {})
            else:
                js = getattr(rf, "json_schema", None)
                schema = js.get("schema", {}) if isinstance(js, dict) else getattr(js, "schema", {})
            if schema and isinstance(schema, dict):
                gen_config["responseSchema"] = _sanitize_schema(schema)

    # After building gen_config, apply model defaults for missing params
    from .models import MODEL_GENERATION_DEFAULTS
    model_defaults = MODEL_GENERATION_DEFAULTS.get(backend_model)
    if model_defaults is None and backend_model.startswith("gemini"):
        model_defaults = {"temperature": 1.0, "topP": 0.95, "topK": 64}
    elif model_defaults is None:
        model_defaults = {}
    for key, default_value in model_defaults.items():
        if key not in gen_config:
            gen_config[key] = default_value

    if gen_config:
        req_payload["generationConfig"] = gen_config

    tool_choice = request.tool_choice
    if tool_choice:
        choice_type = None
        tool_name = None
        if isinstance(tool_choice, str):
            choice_type = tool_choice
        elif isinstance(tool_choice, dict):
            choice_type = tool_choice.get("type")
            tool_name = tool_choice.get("name")
        else:
            choice_type = getattr(tool_choice, "type", None)
            tool_name = getattr(tool_choice, "name", None)

        if choice_type == "auto":
            req_payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        elif choice_type in ("any", "required"):
            req_payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
        elif choice_type == "none":
            req_payload["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        elif choice_type == "tool" and tool_name:
            req_payload["toolConfig"] = {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [tool_name]}
            }

    req_payload["safetySettings"] = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    ]

    full_payload = {
        "project": project_id,
        "model": backend_model,
        "request": req_payload,
    }
    return full_payload, backend_model


def is_known_harness_system_prompt(system_text: str) -> bool:
    """Detect if the system prompt originates from Claude Code or Antigravity harnesses.

    Harnesses rely on exact XML framing, capabilities declarations, and tool use rules.
    Detecting these ensures we never inject interfering synthetic prompts or alter structure.
    """
    if not system_text:
        return False
    lower = system_text.lower()
    return (
        "<claude_info>" in lower
        or "claude code" in lower
        or "<antigravity_instructions>" in lower
        or "<policy_spec>" in lower
        or "you are an agentic coding assistant" in lower
        or "<context>" in lower
        or "<instructions>" in lower
    )


def _join_system_blocks(blocks: list[str]) -> str:
    """Join multiple system prompt text blocks while preserving exact structure, tags, and formatting."""
    if not blocks:
        return ""
    if len(blocks) == 1:
        return blocks[0]

    result: list[str] = []
    for b in blocks:
        if not b:
            continue
        if not result:
            result.append(b)
            continue

        prev = result[-1]
        prev_stripped = prev.rstrip("\r\n")
        curr_stripped = b.lstrip("\r\n")

        prev_trailing_newlines = len(prev) - len(prev_stripped)
        curr_leading_newlines = len(b) - len(curr_stripped)
        existing_newlines = prev_trailing_newlines + curr_leading_newlines

        is_xml_boundary = prev_stripped.endswith(">") and (
            curr_stripped.startswith("<") or curr_stripped.startswith("</")
        )

        if existing_newlines >= 2:
            result[-1] = prev_stripped + "\n\n"
            result.append(curr_stripped)
        elif existing_newlines == 1:
            if is_xml_boundary:
                result[-1] = prev_stripped + "\n"
                result.append(curr_stripped)
            else:
                result[-1] = prev_stripped + "\n\n"
                result.append(curr_stripped)
        else:
            if is_xml_boundary:
                result[-1] = prev_stripped + "\n"
                result.append(curr_stripped)
            else:
                result[-1] = prev_stripped + "\n\n"
                result.append(curr_stripped)

    return "".join(result)


def extract_system_cache_control(system: Any) -> list[dict[str, Any]]:
    """Extract cache_control annotations from system prompt blocks if present."""
    if not isinstance(system, list):
        return []
    markers: list[dict[str, Any]] = []
    for block in system:
        if isinstance(block, dict):
            cc = block.get("cache_control")
            if isinstance(cc, dict):
                markers.append(cc)
        else:
            cc = getattr(block, "cache_control", None)
            if isinstance(cc, dict):
                markers.append(cc)
    return markers


def _extract_system_text(system: Any) -> str:
    """Extract system text while preserving 100% structural fidelity, XML hierarchy, and markdown formatting."""
    if not system:
        return ""
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        blocks_text: list[str] = []
        for block in system:
            if isinstance(block, str):
                if block:
                    blocks_text.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" or not btype:
                    text_val = block.get("text")
                    if text_val is None and "content" in block:
                        text_val = block.get("content")
                    if text_val is not None:
                        blocks_text.append(str(text_val))
                elif btype in ("citation", "reference"):
                    cit_text = block.get("text") or block.get("content") or ""
                    if cit_text:
                        blocks_text.append(str(cit_text))
            else:
                btype = getattr(block, "type", None)
                if btype == "text" or not btype:
                    text_val = getattr(block, "text", None)
                    if text_val is None and hasattr(block, "content"):
                        text_val = getattr(block, "content", None)
                    if text_val is not None:
                        blocks_text.append(str(text_val))
                elif btype in ("citation", "reference"):
                    cit_text = getattr(block, "text", None) or getattr(block, "content", None) or ""
                    if cit_text:
                        blocks_text.append(str(cit_text))

        return _join_system_blocks(blocks_text)

    return ""


# Allowed fields in Google Gemini / Cloud Code Schema protobuf
_GEMINI_ALLOWED_SCHEMA_FIELDS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "items",
}


def _sanitize_schema(schema: Any, root_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sanitize JSON schema to strictly conform to Google Gemini Schema protobuf.

    Strips unsupported fields such as $schema, $defs,
    definitions, propertyNames, patternProperties, etc., and normalizes types.
    Preserves semantically vital fields: additionalProperties, const, default,
    description, and all required properties.
    """
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}

    raw = dict(schema)
    if root_schema is None:
        root_schema = raw

    # Handle $ref by inlining
    if "$ref" in raw and isinstance(raw["$ref"], str) and root_schema:
        ref_path = raw["$ref"]
        curr = root_schema
        if ref_path.startswith("#/"):
            parts = ref_path[2:].split("/")
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    curr = None
                    break
        elif ref_path in root_schema.get("$defs", {}):
            curr = root_schema["$defs"][ref_path]
        elif ref_path in root_schema.get("definitions", {}):
            curr = root_schema["definitions"][ref_path]
        else:
            curr = None

        if isinstance(curr, dict):
            return _sanitize_schema(curr, root_schema)

    result: dict[str, Any] = {}

    # Simplify anyOf / oneOf / allOf if type is not directly present
    nullable = bool(raw.get("nullable", False))
    for combiner in ("anyOf", "oneOf", "allOf"):
        if combiner in raw and isinstance(raw[combiner], list):
            for variant in raw[combiner]:
                if isinstance(variant, dict):
                    if "$ref" in variant and isinstance(variant["$ref"], str) and root_schema:
                        resolved_var = _sanitize_schema(variant, root_schema)
                        if isinstance(resolved_var, dict):
                            variant = resolved_var

                    vtype = variant.get("type")
                    if vtype == "null" or (isinstance(vtype, list) and "null" in vtype):
                        nullable = True
                    elif "type" not in raw and vtype:
                        raw["type"] = vtype
                        if "properties" in variant and "properties" not in raw:
                            raw["properties"] = variant["properties"]
                        if "items" in variant and "items" not in raw:
                            raw["items"] = variant["items"]
                        if "enum" in variant and "enum" not in raw:
                            raw["enum"] = variant["enum"]
                        if "description" in variant and "description" not in raw:
                            raw["description"] = variant["description"]

    # Normalize type
    schema_type = raw.get("type")
    if isinstance(schema_type, list):
        if "null" in schema_type:
            nullable = True
            schema_type = [t for t in schema_type if t != "null"]
        schema_type = schema_type[0] if schema_type else None

    if isinstance(schema_type, str):
        st_upper = schema_type.upper()
        if st_upper in ("OBJECT", "STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY"):
            result["type"] = st_upper
        else:
            result["type"] = "STRING"
    elif "properties" in raw:
        result["type"] = "OBJECT"
    elif "items" in raw:
        result["type"] = "ARRAY"
    elif "enum" in raw:
        result["type"] = "STRING"
    elif "const" in raw and raw["const"] is not None:
        const_val = raw["const"]
        if isinstance(const_val, bool):
            result["type"] = "BOOLEAN"
        elif isinstance(const_val, int):
            result["type"] = "INTEGER"
        elif isinstance(const_val, float):
            result["type"] = "NUMBER"
        else:
            result["type"] = "STRING"
    else:
        result["type"] = "OBJECT"

    if nullable:
        result["nullable"] = True

    if "description" in raw and raw["description"] is not None:
        result["description"] = str(raw["description"])
    elif "title" in raw and raw["title"] is not None:
        result["description"] = str(raw["title"])

    if "default" in raw and raw["default"] is not None:
        default_str = f"Default: {raw['default']}"
        if "description" in result:
            result["description"] = f"{result['description']}\n{default_str}"
        else:
            result["description"] = default_str

    if "const" in raw and raw["const"] is not None:
        result["enum"] = [str(raw["const"])]

    if "format" in raw and isinstance(raw["format"], str):
        result["format"] = raw["format"]

    if "enum" in raw and isinstance(raw["enum"], list):
        result["enum"] = [str(x) for x in raw["enum"]]

    if "maxItems" in raw and isinstance(raw["maxItems"], int):
        result["maxItems"] = raw["maxItems"]

    if "minItems" in raw and isinstance(raw["minItems"], int):
        result["minItems"] = raw["minItems"]

    if "additionalProperties" in raw and isinstance(raw["additionalProperties"], bool):
        result["additionalProperties"] = raw["additionalProperties"]

    # Properties
    sanitized_props: dict[str, Any] = {}
    if "properties" in raw and isinstance(raw["properties"], dict):
        for prop_name, prop_def in raw["properties"].items():
            if isinstance(prop_name, str):
                sanitized_props[prop_name] = _sanitize_schema(prop_def, root_schema)

    # Required: Gemini requires all listed required fields to be declared in properties.
    # If a field is in required but missing from properties, inject a generic string definition
    # so Gemini's protobuf accepts it without stripping the required semantic constraint.
    if "required" in raw and isinstance(raw["required"], list):
        req_list: list[str] = []
        for k in raw["required"]:
            if isinstance(k, str):
                prop_key = str(k)
                req_list.append(prop_key)
                if prop_key not in sanitized_props:
                    sanitized_props[prop_key] = {
                        "type": "STRING",
                        "description": f"Required parameter `{prop_key}`",
                    }
        if req_list:
            result["required"] = req_list

    if sanitized_props:
        result["properties"] = sanitized_props

    # Items for ARRAY schemas
    if "items" in raw:
        if isinstance(raw["items"], dict):
            result["items"] = _sanitize_schema(raw["items"], root_schema)
        elif isinstance(raw["items"], list) and raw["items"]:
            result["items"] = _sanitize_schema(raw["items"][0], root_schema)
    elif result.get("type") == "ARRAY":
        result["items"] = {"type": "STRING"}

    return result



def _build_contents(
    messages: list[Any],
    signature_cache: dict[str, str],
    backend_model: str = "",
) -> list[dict[str, Any]]:
    is_gemini = backend_model.startswith("gemini")
    valid_tool_call_ids: set[str] = set()

    # First pass: look for any thinking signatures and tool signatures in messages
    for msg in messages:
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, list):
            for block in content:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else ""
                )
                tid = getattr(block, "id", None) or (
                    block.get("id") if isinstance(block, dict) else ""
                )
                sig = getattr(block, "signature", None) or (
                    block.get("signature") if isinstance(block, dict) else ""
                )
                if btype == "thinking" and sig:
                    signature_cache["latest_turn_sig"] = sig
                elif btype == "tool_use":
                    if not sig and tid:
                        sig = signature_cache.get(tid)
                    if sig:
                        valid_tool_call_ids.add(tid)
                        if tid and tid not in signature_cache:
                            signature_cache[tid] = sig

    raw_turns: list[dict[str, Any]] = []

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
                elif btype == "thinking":
                    thinking_val = getattr(block, "thinking", None) or (
                        block.get("thinking") if isinstance(block, dict) else ""
                    )
                    sig = getattr(block, "signature", None) or (
                        block.get("signature") if isinstance(block, dict) else ""
                    )
                    if sig:
                        signature_cache["latest_turn_sig"] = sig
                    if thinking_val:
                        thought_part: dict[str, Any] = {"text": thinking_val, "thought": True}
                        if sig:
                            thought_part["thoughtSignature"] = sig
                        parts.append(thought_part)
                elif btype in ("image", "document"):
                    source = getattr(block, "source", None) or (
                        block.get("source") if isinstance(block, dict) else {}
                    )
                    data_val = (
                        source.get("data")
                        if isinstance(source, dict)
                        else getattr(source, "data", None)
                    )
                    default_mime = "application/pdf" if btype == "document" else "image/jpeg"
                    media_type = (
                        source.get("media_type")
                        if isinstance(source, dict)
                        else getattr(source, "media_type", default_mime)
                    )
                    if data_val:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": media_type or default_mime,
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

                    sig = (
                        getattr(block, "signature", None)
                        or (block.get("signature") if isinstance(block, dict) else "")
                        or signature_cache.get(tool_id)
                    )

                    part: dict[str, Any] = {
                        "functionCall": {
                            "name": tool_name,
                            "args": tool_input,
                            "id": tool_id,
                        }
                    }
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
                    is_error = getattr(block, "is_error", False) or (
                        block.get("is_error") if isinstance(block, dict) else False
                    )

                    text_items: list[str] = []
                    media_parts: list[dict[str, Any]] = []

                    if isinstance(result_content, list):
                        for c in result_content:
                            c_type = getattr(c, "type", None) or (
                                c.get("type") if isinstance(c, dict) else ""
                            )
                            if c_type == "text" or not c_type:
                                t = getattr(c, "text", None) or (
                                    c.get("text") if isinstance(c, dict) else ""
                                )
                                if t:
                                    text_items.append(str(t))
                            elif c_type in ("image", "document"):
                                src = getattr(c, "source", None) or (
                                    c.get("source") if isinstance(c, dict) else {}
                                )
                                d = (
                                    src.get("data")
                                    if isinstance(src, dict)
                                    else getattr(src, "data", None)
                                )
                                m = (
                                    src.get("media_type")
                                    if isinstance(src, dict)
                                    else getattr(src, "media_type", None)
                                )
                                if not m:
                                    m = "application/pdf" if c_type == "document" else "image/jpeg"
                                if d:
                                    media_parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": m,
                                                "data": d,
                                            }
                                        }
                                    )
                        result_str = "\n".join(filter(None, text_items))
                    else:
                        result_str = str(result_content)

                    if is_error:
                        if not result_str.startswith("Error:") and not result_str.startswith("[ERROR]"):
                            result_str = f"Error: {result_str}"

                    resp_payload: dict[str, Any] = {"content": result_str}
                    if is_error:
                        resp_payload["error"] = True

                    part = {
                        "functionResponse": {
                            "name": _find_tool_name_by_id(
                                messages, tool_use_id, signature_cache
                            )
                            or "tool",
                            "response": resp_payload,
                            "id": tool_use_id or f"toolu_{uuid.uuid4().hex[:20]}",
                        }
                    }
                    parts.append(part)
                    parts.extend(media_parts)
        if parts:
            raw_turns.append({"role": gemini_role, "parts": parts})

    # Merge consecutive turns with identical role into one turn
    merged: list[dict[str, Any]] = []
    for turn in raw_turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["parts"].extend(turn["parts"])
        else:
            merged.append(turn)

    # Antigravity streamGenerateContent API requires first turn to be 'user' and last turn to be 'user'
    if merged:
        if merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "parts": [{"text": " "}]})
        if merged[-1]["role"] == "model":
            merged.append({"role": "user", "parts": [{"text": " "}]})

    return merged


def _find_tool_name_by_id(
    messages: list[Any],
    tool_id: str,
    signature_cache: dict[str, str] | None = None,
) -> str | None:
    if tool_id:
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

    # Fallback: match against the most recent tool_use in the conversation history
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, list):
            for block in reversed(content):
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else ""
                )
                bname = getattr(block, "name", None) or (
                    block.get("name") if isinstance(block, dict) else ""
                )
                if btype == "tool_use" and bname:
                    return bname

    return None


class AntigravityStreamTranslator:
    """Translates SSE streaming lines from streamGenerateContent to Anthropic SSE events."""

    def __init__(
        self,
        public_model: str,
        signature_cache: Any,
        *,
        input_tokens: int = 0,
    ) -> None:
        self.public_model = public_model
        self.signature_cache = signature_cache
        self.input_tokens = input_tokens
        self.cache_read_tokens = 0
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._message_started = False
        self._current_block_index = -1
        self._current_block_type: str | None = None
        self._total_output_tokens = 0
        self._stop_reason = "end_turn"
        self._pending_sig: str | None = None
        self._json_buffer = ""

    def process_line(self, line: str) -> list[str]:
        """Process one SSE line and return zero or more Anthropic SSE event frames."""
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return []

        payload_str = stripped[5:].strip()
        if not payload_str:
            return []

        if self._json_buffer:
            self._json_buffer += "\n" + payload_str
        else:
            self._json_buffer = payload_str

        # Protect against unbounded stream buffer OOM
        if len(self._json_buffer) > 5 * 1024 * 1024:
            self._json_buffer = ""
            raise ValueError("Antigravity SSE JSON buffer exceeded 5MB limit without valid JSON")

        try:
            chunk = json.loads(self._json_buffer)
            self._json_buffer = ""
        except json.JSONDecodeError:
            # wait for more
            return []

        events: list[str] = []

        resp = chunk.get("response", {})
        candidates = resp.get("candidates", [])
        usage_meta = resp.get("usageMetadata", {})
        if usage_meta:
            candidates_tokens = usage_meta.get("candidatesTokenCount")
            if isinstance(candidates_tokens, int):
                self._total_output_tokens = candidates_tokens
            prompt_tokens = usage_meta.get("promptTokenCount")
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                self.input_tokens = prompt_tokens
            cached_tokens = usage_meta.get("cachedContentTokenCount")
            if isinstance(cached_tokens, int) and cached_tokens > 0:
                self.cache_read_tokens = cached_tokens

        # Start message on first valid payload
        if not self._message_started:
            self._message_started = True
            msg_usage: dict[str, Any] = {
                "input_tokens": self.input_tokens,
                "output_tokens": 1,
            }
            if self.cache_read_tokens > 0:
                msg_usage["cache_read_input_tokens"] = self.cache_read_tokens
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
                            "usage": msg_usage,
                        },
                    },
                )
            )

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
                    self._pending_sig = sig

                is_thought = bool(part.get("thought", False))
                text = part.get("text")

                if is_thought:
                    # Thinking block
                    if self._current_block_type != "thinking":
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
                        self._current_block_type = "thinking"
                        events.append(
                            format_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": self._current_block_index,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                },
                            )
                        )
                    if text:
                        events.append(
                            format_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": self._current_block_index,
                                    "delta": {"type": "thinking_delta", "thinking": text},
                                },
                            )
                        )
                    if sig:
                        events.append(
                            format_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": self._current_block_index,
                                    "delta": {"type": "signature_delta", "signature": sig},
                                },
                            )
                        )
                elif text is not None and len(text) > 0:
                    # Text delta
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

                    effective_sig = (
                        (part.get("thoughtSignature") if isinstance(part.get("thoughtSignature"), str) else None)
                        or sig
                        or self._pending_sig
                    )

                    # If this part or stream has thoughtSignature, record it for this tool call ID
                    self.signature_cache[f"name_{call_id}"] = fn_name
                    if effective_sig:
                        self.signature_cache[call_id] = effective_sig
                        self.signature_cache[fn_name] = effective_sig

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
                    
                    full_args = json.dumps(fn_args)
                    chunk_size = 1024
                    for i in range(0, len(full_args), chunk_size):
                        events.append(
                            format_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": self._current_block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": full_args[i:i + chunk_size],
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

        usage_dict: dict[str, Any] = {
            "output_tokens": max(1, self._total_output_tokens),
        }
        if self.cache_read_tokens > 0:
            usage_dict["cache_read_input_tokens"] = self.cache_read_tokens

        events.append(
            format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": usage_dict,
                },
            )
        )
        events.append(format_sse_event("message_stop", {"type": "message_stop"}))
        return events
