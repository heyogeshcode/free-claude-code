"""Universal Bidirectional Schema Normalizer for Anthropic Messages and OpenAI Chat Completions formats."""

import json
import uuid
from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest


class NvidiaSchemaNormalizer:
    """Bidirectional schema translation between Anthropic Messages and OpenAI Chat / NIM formats."""

    @staticmethod
    def anthropic_to_openai_chat(
        request: MessagesRequest | dict[str, Any],
        *,
        stream: bool = True,
        default_model: str = "meta/llama-3.3-70b-instruct",
    ) -> dict[str, Any]:
        """Translate Anthropic Messages request payload to OpenAI Chat Completions format."""
        if isinstance(request, MessagesRequest):
            raw = request.model_dump(exclude_unset=True)
        else:
            raw = dict(request)

        model = raw.get("model") or default_model
        # Strip provider prefix if present (e.g. "nvidia_nim/meta/llama..." -> "meta/llama...")
        if "/" in model:
            parts = model.split("/", 1)
            if parts[0] in ("nvidia_nim", "nvidia_fallback", "anthropic"):
                model = parts[1]
        elif model.startswith("claude-"):
            model = default_model

        openai_messages: list[dict[str, Any]] = []

        # 1. Process System Prompts
        system_content = raw.get("system")
        if system_content:
            if isinstance(system_content, str):
                openai_messages.append({"role": "system", "content": system_content})
            elif isinstance(system_content, list):
                sys_parts = []
                for item in system_content:
                    if isinstance(item, str):
                        sys_parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        sys_parts.append(item.get("text", ""))
                if sys_parts:
                    openai_messages.append(
                        {"role": "system", "content": "\n\n".join(sys_parts)}
                    )

        # 2. Process Messages
        for msg in raw.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                text_parts = []
                tool_calls = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    b_type = block.get("type")
                    if b_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif b_type == "thinking":
                        # Preserve thinking block as XML tag for models that support it
                        thinking = block.get("thinking", "")
                        text_parts.append(f"<thinking>{thinking}</thinking>")
                    elif b_type == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )
                    elif b_type == "tool_result":
                        # Anthropic puts tool_result in user message; OpenAI uses role: "tool"
                        res_content = block.get("content", "")
                        if isinstance(res_content, list):
                            res_parts = [
                                b.get("text", "")
                                for b in res_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            res_content = "\n".join(res_parts)
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(res_content),
                            }
                        )

                if role == "assistant" and tool_calls:
                    msg_dict: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                        "tool_calls": tool_calls,
                    }
                    openai_messages.append(msg_dict)
                elif text_parts:
                    openai_messages.append(
                        {"role": role, "content": "\n".join(text_parts)}
                    )

        openai_body: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "stream": stream,
        }

        # 3. Process Tools
        tools = raw.get("tools")
        if tools and isinstance(tools, list):
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
                if isinstance(t, dict)
            ]
            if openai_tools:
                openai_body["tools"] = openai_tools

        # 4. Map hyperparameters
        if "max_tokens" in raw:
            openai_body["max_tokens"] = raw["max_tokens"]
        if "temperature" in raw and raw["temperature"] is not None:
            openai_body["temperature"] = raw["temperature"]
        if "top_p" in raw and raw["top_p"] is not None:
            openai_body["top_p"] = raw["top_p"]
        if raw.get("stop_sequences"):
            openai_body["stop"] = raw["stop_sequences"]

        return openai_body

    @staticmethod
    def openai_chunk_to_anthropic_events(
        chunk_data: dict[str, Any],
        state: dict[str, Any],
        *,
        model: str,
        message_id: str,
    ) -> list[str]:
        """Convert one OpenAI streaming chunk dictionary to a sequence of Anthropic SSE lines.

        State tracks active blocks:
        state = {
            "started": bool,
            "current_block_index": int,
            "block_type": "text" | "tool_use" | "thinking" | None,
            "tool_id": str | None,
            "tool_name": str | None,
            "has_tool": bool,
        }
        """
        events: list[str] = []

        # 1. Emit message_start if not started
        if not state.get("started"):
            state["started"] = True
            state["current_block_index"] = -1
            state["block_type"] = None
            state["has_tool"] = False

            start_payload = {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            events.append(
                f"event: message_start\ndata: {json.dumps(start_payload)}\n\n"
            )

        choices = chunk_data.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # 2. Check for reasoning content (DeepSeek R1 / NIM reasoning)
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning_delta:
            if state["block_type"] != "thinking":
                if state["block_type"] is not None:
                    # Close previous block
                    stop_ev = {
                        "type": "content_block_stop",
                        "index": state["current_block_index"],
                    }
                    events.append(
                        f"event: content_block_stop\ndata: {json.dumps(stop_ev)}\n\n"
                    )
                state["current_block_index"] += 1
                state["block_type"] = "thinking"
                start_ev = {
                    "type": "content_block_start",
                    "index": state["current_block_index"],
                    "content_block": {"type": "thinking", "thinking": ""},
                }
                events.append(
                    f"event: content_block_start\ndata: {json.dumps(start_ev)}\n\n"
                )

            delta_ev = {
                "type": "content_block_delta",
                "index": state["current_block_index"],
                "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
            }
            events.append(
                f"event: content_block_delta\ndata: {json.dumps(delta_ev)}\n\n"
            )

        # 3. Check for text content
        text_delta = delta.get("content")
        if text_delta:
            if state["block_type"] != "text":
                if state["block_type"] is not None:
                    stop_ev = {
                        "type": "content_block_stop",
                        "index": state["current_block_index"],
                    }
                    events.append(
                        f"event: content_block_stop\ndata: {json.dumps(stop_ev)}\n\n"
                    )
                state["current_block_index"] += 1
                state["block_type"] = "text"
                start_ev = {
                    "type": "content_block_start",
                    "index": state["current_block_index"],
                    "content_block": {"type": "text", "text": ""},
                }
                events.append(
                    f"event: content_block_start\ndata: {json.dumps(start_ev)}\n\n"
                )

            delta_ev = {
                "type": "content_block_delta",
                "index": state["current_block_index"],
                "delta": {"type": "text_delta", "text": text_delta},
            }
            events.append(
                f"event: content_block_delta\ndata: {json.dumps(delta_ev)}\n\n"
            )

        # 4. Check for tool calls
        tool_calls = delta.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            state["has_tool"] = True
            for tc in tool_calls:
                fn = tc.get("function", {})
                tc_id = tc.get("id")
                fn_name = fn.get("name")
                fn_args = fn.get("arguments")

                if tc_id or fn_name:
                    if state["block_type"] is not None:
                        stop_ev = {
                            "type": "content_block_stop",
                            "index": state["current_block_index"],
                        }
                        events.append(
                            f"event: content_block_stop\ndata: {json.dumps(stop_ev)}\n\n"
                        )
                    state["current_block_index"] += 1
                    state["block_type"] = "tool_use"
                    state["tool_id"] = tc_id or f"call_{uuid.uuid4().hex[:8]}"
                    state["tool_name"] = fn_name or ""
                    start_ev = {
                        "type": "content_block_start",
                        "index": state["current_block_index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": state["tool_id"],
                            "name": state["tool_name"],
                            "input": {},
                        },
                    }
                    events.append(
                        f"event: content_block_start\ndata: {json.dumps(start_ev)}\n\n"
                    )

                if fn_args:
                    delta_ev = {
                        "type": "content_block_delta",
                        "index": state["current_block_index"],
                        "delta": {"type": "input_json_delta", "partial_json": fn_args},
                    }
                    events.append(
                        f"event: content_block_delta\ndata: {json.dumps(delta_ev)}\n\n"
                    )

        # 5. Handle finish_reason
        if finish_reason:
            if state["block_type"] is not None:
                stop_ev = {
                    "type": "content_block_stop",
                    "index": state["current_block_index"],
                }
                events.append(
                    f"event: content_block_stop\ndata: {json.dumps(stop_ev)}\n\n"
                )
                state["block_type"] = None

            stop_reason = "end_turn"
            if finish_reason in ("tool_calls", "function_call") or state.get(
                "has_tool"
            ):
                stop_reason = "tool_use"
            elif finish_reason == "length":
                stop_reason = "max_tokens"

            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 1},
            }
            events.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n")

            msg_stop = {"type": "message_stop"}
            events.append(f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n")

        return events
