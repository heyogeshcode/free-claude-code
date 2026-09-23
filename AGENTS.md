# 🔧 Free Claude Code Proxy — Deep Optimization Plan

> **Goal**: Eliminate model intelligence degradation, hallucination, and quality loss when accessing Antigravity models (Gemini, Claude) through this proxy, making the proxy experience indistinguishable from the native Antigravity CLI.
>
> **Constraint**: Do NOT edit the running version at `~/.local/bin`. Only edit files in THIS repository (`~/Downloads/free-claude-code`).
>
> **Execution Strategy**: Use **sub-agents** for independent phases. Each phase below is designed to be parallelizable where marked.

---

## 📋 Root Cause Analysis (Completed by Diagnostic Agent)

After exhaustive analysis of 25+ source files and ~6000 lines of the proxy codebase, the following **root causes of model quality degradation** have been identified:

### 🔴 Critical Issues (Directly Cause Hallucination/Quality Loss)

1. **Lossy System Prompt Translation** (`translator.py` L86-101)
   - `_extract_system_text()` joins multi-block system prompts with `\n\n`, destroying structural hierarchy, XML tags, and rich formatting that harnesses like Antigravity CLI send.
   - This is the **#1 cause of hallucination** — when the system prompt structure is destroyed, the model loses critical context about its role, capabilities, and constraints.

2. **Thinking/Reasoning Budget Truncation** (`core/reasoning.py` L34-41)
   - The `_EFFORT_BUDGET_TOKENS` mapping is extremely low: `HIGH=2048`, `MAX=8192`.
   - Native Antigravity CLI allows much larger thinking budgets (32K-128K). This proxy artificially constrains reasoning capacity, making models appear "dumber."

3. **Tool Schema Lossy Sanitization** (`translator.py` L119-223)
   - `_sanitize_schema()` aggressively strips JSON Schema fields that models rely on: `additionalProperties`, `default`, `const`, `title`, `$defs`, `definitions`, `patternProperties`, `propertyNames`, `if/then/else`.
   - These stripped fields carry semantic meaning. When removed, tool call arguments become ambiguous, causing the model to hallucinate tool inputs.

4. **Forced Conversation Structure Injection** (`translator.py` L401-407)
   - Lines 403-406 inject synthetic `"Hello"` and `"Please continue."` messages when the conversation doesn't start/end with a user turn.
   - These injected messages **confuse the model** by introducing context that the client never intended. Native Antigravity CLI handles turn ordering properly server-side.

5. **Tool Call Fallback to Text Representation** (`translator.py` L333-337)
   - When `thoughtSignature` is missing for Gemini models, tool calls are converted to plain text: `"Calling tool 'name' with arguments: {...}"`.
   - This breaks the tool-calling contract entirely. The model then interprets these as natural language rather than structured tool invocations, causing it to hallucinate tool results.

6. **Tool Result Fallback to Text** (`translator.py` L370-376)
   - Matching issue: tool results without valid tool call IDs become plain text `"[Tool 'name' result]:\n..."`.
   - This destroys the functional calling chain, causing models to lose track of tool execution state.


19. **Silent PDF/Document Modality Drop** (`translator.py` L376)
    - Anthropic API supports `type="document"` (base64 PDFs). `_build_contents()` completely ignores `btype == "document"`.
    - *Impact*: PDF attachments in Claude Code are silently stripped. The model receives an empty context and massively hallucinates document contents.

20. **Context Caching Complete Loss** (`translator.py`)
    - Anthropic uses `cache_control: {"type": "ephemeral"}` for prompt caching. Gemini uses `cachedContent`.
    - The proxy strips all `cache_control` annotations.
    - *Impact*: Defeats Claude Code's caching optimizations. Causes 100k+ token system prompts to be reprocessed every single turn, spiking API costs and hitting rate limits instantly.

### 🟡 Significant Issues (Degrade Quality)

7. **Missing `thinking` Block Handling in Content Translation** (`translator.py` L226-408)
   - `_build_contents()` completely ignores `thinking` content blocks from previous turns.
   - When multi-turn conversations include reasoning history, this context is silently dropped, making the model "forget" its previous reasoning chains.

8. **Temperature Not Properly Forwarded** (`translator.py` L67)
   - Temperature is forwarded only when explicitly set. The proxy doesn't send the model-specific optimal defaults that native Antigravity CLI uses.

9. **`top_p` and `top_k` Not Forwarded** (`translator.py` L62-76)
   - The generation config builder only handles `maxOutputTokens`, `temperature`, and `thinkingConfig`. Parameters `top_p`, `top_k`, and `stop_sequences` are silently dropped.

10. **`max_tokens` Incorrectly Forwarded** (`translator.py` L64-65)
    - `request.max_tokens` is used as-is. However, for thinking-enabled models, `max_tokens` in Anthropic API includes both thinking AND output tokens. The proxy should split this budget appropriately for the `maxOutputTokens` vs `thinkingBudget` fields in Gemini API.

11. **SSE Stream Chunking Can Corrupt Multi-byte Characters** (`provider.py` L337)
    - `response.aiter_lines()` may split on byte boundaries rather than character boundaries for multi-byte UTF-8 content, potentially corrupting SSE payloads.

12. **Signature Cache Unbounded Growth** (`provider.py` L86-88)
    - Cache trim at 2000 entries uses simple slice, potentially evicting still-needed signatures during long sessions, causing `thoughtSignature` mismatches and 400 errors.

13. **Missing `stop_sequences` Translation** (`translator.py`)
    - Anthropic `stop_sequences` are never translated to Gemini `stopSequences` in the generation config. This means the model doesn't know when to stop generating, leading to verbose or off-topic output.

14. **`tool_choice` Not Forwarded** (`translator.py`)
    - Anthropic `tool_choice` (auto/any/specific) is never translated to Gemini `toolConfig`. The model defaults to its own tool selection strategy, which may differ from what the client requested.


21. **Required Tool Arguments Lost** (`translator.py` L278-289)
    - `_sanitize_schema()` drops required properties from the `required` list if they aren't explicitly defined in `properties` (which is valid in JSON Schema if `additionalProperties: true`).
    - *Impact*: The model thinks critical arguments are optional and omits them, leading to tool validation errors.

22. **Token Usage Discarding** (`translator.py` L578-582, L780)
    - `process_line()` parses `usageMetadata` but ignores `promptTokenCount`. `finalize()` emits `input_tokens: 0` (or the default).
    - *Impact*: Breaks token usage tracking, billing, and context window limits in Claude Code/OpenCode.

23. **Multimedia Tool Results Stripped** (`translator.py` L428-445)
    - `btype == "tool_result"` only extracts text. If a tool returns an image (e.g. browser screenshots), the image is silently discarded.
    - *Impact*: Multimodal tools (like browser automation or UI rendering) are entirely broken.

24. **Tool Error Semantics Lost** (`translator.py` L428)
    - Anthropic tool results include `is_error=True`. The translator ignores this boolean and just concatenates the text.
    - *Impact*: The model loses explicit error signaling, relying entirely on natural language parsing to figure out a tool failed.


25. **Unbounded Stream Buffer / OOM Vulnerability** (`translator.py` L537-548)
    - `AntigravityStreamTranslator.process_line()` appends to `_json_buffer` infinitely if `json.loads` fails.
    - *Impact*: A malformed or malicious SSE stream from upstream will cause the proxy to consume memory infinitely until the process crashes (OOM), destroying all active sessions.

### 🟢 Minor Issues (Polish)

15. **Empty Content Fallback** (`translator.py` L58-59)
    - Empty messages get a single empty text part `""`. Better to omit entirely or use a meaningful placeholder.

16. **Token Count Estimation** (`core/token_estimation.py`)
    - Input token estimates are rough. While not directly causing hallucination, inaccurate counts can lead to premature truncation of context.

17. **Progress Timeout Too Aggressive** (`execution.py` L68-75)
    - For complex thinking tasks, the progress timeout may fire before the model finishes its reasoning phase (especially for large thinking budgets).

18. **Missing `response_format` Translation** (`translator.py`)
    - Anthropic `response_format` (JSON mode) is not translated to Gemini `responseMimeType`, causing structured output requests to fail or produce unstructured output.

---

## 🏗️ Implementation Phases

### Phase 1: System Prompt Fidelity (CRITICAL — Anti-Hallucination)

**Sub-agent**: `prompt-fidelity-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Rewrite `_extract_system_text()`** to preserve structural integrity:
   - Maintain XML tags, markdown formatting, and hierarchical structure
   - Preserve `<type>`, `<instructions>`, `<context>`, and other structural elements that harnesses like Antigravity use
   - Handle `cache_control` annotations properly (Anthropic's prompt caching markers)
   - Preserve citation markers and reference blocks
   - **Test**: Verify that a complex system prompt with XML tags, markdown headers, and nested blocks is preserved character-for-character

2. **Add a system prompt enhancement layer** for known harness patterns:
   - Detect when the system prompt comes from Claude Code / Antigravity harness
   - Preserve the exact prompt structure that gives models their "intelligence"
   - Don't add any extra system text that could confuse the model

---

### Phase 2: Thinking/Reasoning Budget Fix (CRITICAL — Intelligence)

**Sub-agent**: `reasoning-budget-agent`

**Files to modify**:
- `src/free_claude_code/core/reasoning.py`
- `src/free_claude_code/providers/antigravity/translator.py`
- `src/free_claude_code/application/reasoning.py`

**Tasks**:

1. **Increase `_EFFORT_BUDGET_TOKENS` to match native Antigravity CLI**:
   ```python
   _EFFORT_BUDGET_TOKENS = {
       ReasoningEffort.MINIMAL: 1_024,
       ReasoningEffort.LOW: 4_096,
       ReasoningEffort.MEDIUM: 10_240,
       ReasoningEffort.HIGH: 32_768,
       ReasoningEffort.XHIGH: 65_536,
       ReasoningEffort.MAX: 131_072,
   }
   ```

2. **Fix `max_tokens` budget splitting** in `translate_messages_request()`:
   - When thinking is enabled, properly partition the `max_tokens` budget between `maxOutputTokens` and `thinkingBudget`
   - Ensure the total doesn't exceed the model's actual maximum
   - For Gemini models: `maxOutputTokens` should be `max_tokens - thinkingBudget`
   - For Claude-on-Antigravity: pass through natively

3. **Properly forward `budget_tokens` from client request**:
   - When the client explicitly sets `thinking.budget_tokens`, use that value exactly
   - Don't clamp it to the effort mapping

---

### Phase 3: Tool Calling Fidelity (CRITICAL — Anti-Hallucination)

**Sub-agent**: `tool-calling-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Rewrite `_sanitize_schema()` to preserve semantically important fields**:
   - Keep `additionalProperties` — this tells the model whether extra fields are allowed
   - Keep `default` values — models use these to understand optional parameters
   - Keep `const` — constrains parameter values
   - Keep `title` as `description` fallback — provides human-readable context
   - Properly handle `$defs`/`definitions` by inlining referenced schemas
   - Preserve `oneOf`/`anyOf`/`allOf` structure more faithfully instead of crude flattening

2. **Fix tool call signature handling** (the text fallback problem):
   - Instead of falling back to text representation when `thoughtSignature` is missing, generate a valid placeholder signature or use a signature-free function call format
   - If Gemini strictly requires signatures, implement proper signature generation/caching strategy
   - Never convert structured tool calls to plain text — this is the worst quality degradation

3. **Fix tool result handling** for missing tool IDs:
   - Maintain a robust tool-call-to-ID mapping that survives across turns
   - When a tool result can't be matched, use the most recent unmatched tool call rather than falling back to text

4. **Forward `tool_choice`** to Gemini's `toolConfig`:
   ```python
   # Translate Anthropic tool_choice to Gemini toolConfig
   if request.tool_choice:
       if request.tool_choice == "auto":
           tool_config = {"functionCallingConfig": {"mode": "AUTO"}}
       elif request.tool_choice == "any":
           tool_config = {"functionCallingConfig": {"mode": "ANY"}}
       elif isinstance(request.tool_choice, dict) and request.tool_choice.get("type") == "tool":
           tool_config = {"functionCallingConfig": {
               "mode": "ANY",
               "allowedFunctionNames": [request.tool_choice["name"]]
           }}
       req_payload["toolConfig"] = tool_config
   ```

---

### Phase 4: Conversation Structure Integrity (CRITICAL — Anti-Hallucination)

**Sub-agent**: `conversation-structure-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Remove synthetic message injection** (lines 403-406):
   - Remove the `"Hello"` and `"Please continue."` synthetic messages entirely
   - Instead, handle the Gemini API's turn-ordering requirement at the API level:
     - If the first turn is `model`, prepend an empty user turn with just a space `" "`
     - If the last turn is `model`, append a minimal user turn with `" "`
   - Use `" "` instead of `"Hello"` / `"Please continue."` — these are semantically loaded words that add unwanted context

2. **Properly handle `thinking` content blocks in conversation history**:
   - When building contents from previous turns, include thinking blocks as `thought: true` parts
   - This preserves the model's reasoning chain across turns, which is critical for multi-step tasks
   - Map Anthropic `thinking` blocks to Gemini `thought` parts

3. **Preserve `signature` and `signature_delta` across turns**:
   - Ensure that `thoughtSignature` from previous thinking blocks is properly cached and reused
   - Fix the signature cache key strategy to avoid collisions in long conversations

---

### Phase 5: Generation Config Completeness (SIGNIFICANT — Quality)

**Sub-agent**: `generation-config-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Forward ALL relevant generation parameters**:
   ```python
   # In translate_messages_request():
   gen_config = {}
   
   if request.max_tokens:
       gen_config["maxOutputTokens"] = request.max_tokens
   if request.temperature is not None:
       gen_config["temperature"] = request.temperature
   if hasattr(request, "top_p") and request.top_p is not None:
       gen_config["topP"] = request.top_p
   if hasattr(request, "top_k") and request.top_k is not None:
       gen_config["topK"] = request.top_k
   
   # Stop sequences
   if request.stop_sequences:
       gen_config["stopSequences"] = list(request.stop_sequences)
   
   # Response format / JSON mode
   if hasattr(request, "response_format"):
       rf = request.response_format
       if isinstance(rf, dict) and rf.get("type") == "json_object":
           gen_config["responseMimeType"] = "application/json"
       elif isinstance(rf, dict) and rf.get("type") == "json_schema":
           gen_config["responseMimeType"] = "application/json"
           if "json_schema" in rf:
               gen_config["responseSchema"] = _sanitize_schema(rf["json_schema"].get("schema", {}))
   ```

2. **Add safety settings bypass for code generation**:
   - Antigravity's native CLI uses lenient safety settings for code generation tasks
   - Add `safetySettings` to the request payload with `BLOCK_NONE` for code-related categories
   - This prevents the model from refusing to generate code that contains security-related patterns

---

### Phase 6: SSE Stream Translation Robustness (SIGNIFICANT — Reliability)

**Sub-agent**: `stream-robustness-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`
- `src/free_claude_code/providers/antigravity/provider.py`

**Tasks**:

1. **Fix SSE line processing robustness** in `AntigravityStreamTranslator.process_line()`:
   - Handle multi-line SSE data fields (data split across multiple `data:` lines)
   - Handle SSE comments (lines starting with `:`)
   - Handle SSE retry fields
   - Buffer incomplete JSON across multiple SSE lines

2. **Fix stream decoding** in `_run_messages()`:
   - Use `response.aiter_bytes()` with a proper UTF-8 streaming decoder instead of `aiter_lines()` to prevent multi-byte character corruption
   - Implement a robust SSE parser that handles edge cases:
     ```python
     async def _parse_sse_stream(response):
         buffer = ""
         async for chunk in response.aiter_text():
             buffer += chunk
             while "\n" in buffer:
                 line, buffer = buffer.split("\n", 1)
                 yield line.rstrip("\r")
     ```

3. **Add stream integrity validation**:
   - Verify that `message_start` is always emitted before `content_block_start`
   - Verify that `content_block_stop` always matches a `content_block_start`
   - Verify that `message_delta` and `message_stop` are emitted exactly once
   - Add recovery for malformed upstream streams

4. **Implement partial JSON accumulation for tool calls**:
   - Currently, `input_json_delta` emits the entire tool arguments in one shot (`json.dumps(fn_args)`)
   - This should be chunked to match how Anthropic streams tool arguments, preventing buffer overflows in downstream clients
   - Split large JSON arguments into ~1KB chunks

---

### Phase 7: Signature Cache Hardening (SIGNIFICANT — Error Prevention)

**Sub-agent**: `signature-cache-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/provider.py`
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Implement LRU-based signature cache** instead of simple dict truncation:
   ```python
   from collections import OrderedDict
   
   class SignatureCache:
       def __init__(self, max_size: int = 5000):
           self._cache: OrderedDict[str, str] = OrderedDict()
           self._max_size = max_size
       
       def get(self, key: str) -> str | None:
           if key in self._cache:
               self._cache.move_to_end(key)
               return self._cache[key]
           return None
       
       def set(self, key: str, value: str) -> None:
           if key in self._cache:
               self._cache.move_to_end(key)
           self._cache[key] = value
           while len(self._cache) > self._max_size:
               self._cache.popitem(last=False)
   ```

2. **Add signature persistence with atomic writes**:
   - Use file-locking for concurrent access safety
   - Add version tracking to detect stale cache

3. **Add signature validation**:
   - Before using a cached signature, verify it hasn't been invalidated
   - Add signature mismatch detection and graceful recovery

---

### Phase 8: Responses API (OpenAI format) Translation Fix (SIGNIFICANT)

**Sub-agent**: `responses-api-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/provider.py`

**Tasks**:

1. **Fix `_sanitize_responses_request()` over-sanitization**:
   - The function strips too many fields from the OpenAI Responses format
   - Preserve `reasoning.summary` when it's a valid value (`"auto"` or `"concise"`)
   - Preserve `truncation` when the value is `"auto"` (don't always strip it)
   - Preserve more content block types in input sanitization

2. **Fix `stream_responses()` double-translation**:
   - Currently the flow is: Responses → Messages → Antigravity → SSE → Messages SSE → Responses SSE
   - This double translation (Messages↔Responses twice) loses information
   - Optimize the pipeline to reduce translation layers

---

### Phase 9: Error Handling & Recovery Enhancement (QUALITY)

**Sub-agent**: `error-handling-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/provider.py`
- `src/free_claude_code/providers/failure_policy.py`

**Tasks**:

1. **Add Antigravity-specific error classification**:
   - Parse Antigravity's specific error response formats
   - Distinguish between quota errors, model-not-found, and temporary failures
   - Add proper retry-after header parsing

2. **Add more granular retry logic**:
   - For `thoughtSignature` mismatch errors (400), retry with cleared signature cache
   - For quota errors (429), implement exponential backoff with jitter
   - For model-not-found (404), try alias resolution before failing

3. **Add stream recovery for partial completions**:
   - When a stream is interrupted mid-content, save the partial content
   - On retry, include the partial content as context to avoid regeneration
   - Implement the `RecoveryController` pattern from `stream_recovery.py` for Antigravity specifically

4. **Improve timeout handling for thinking models**:
   - Increase `progress_timeout_seconds` dynamically when thinking is enabled
   - Detect the "thinking phase" vs "generation phase" and apply different timeouts

---

### Phase 10: Model-Specific Optimization (QUALITY)

**Sub-agent**: `model-optimization-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/models.py`
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Add model-specific generation config defaults**:
   ```python
   MODEL_GENERATION_DEFAULTS = {
       "gemini-3.8-flash-tiered": {
           "temperature": 1.0,  # Gemini works best with temperature 1.0
           "topP": 0.95,
           "topK": 64,
       },
       "claude-sonnet-4-6": {
           "temperature": 1.0,  # Claude's default
       },
       "claude-opus-4-6-thinking": {
           "temperature": 1.0,
       },
   }
   ```

2. **Add model-specific thinking budget defaults**:
   - Gemini models: Use `thinkingBudget` from the thinking config
   - Claude models on Antigravity: Forward thinking parameters natively

3. **Fix context window metadata** for accurate truncation:
   - Update `DEFAULT_ANTIGRAVITY_MODELS` with correct max_output_tokens for each model
   - Add `supports_system_instruction` flag per model
   - Add `supports_tool_calling` flag per model

4. **Add proper Claude model parameter passthrough**:
   - When the backend model is Claude (claude-sonnet-4-6, claude-opus-4-6-thinking), certain Anthropic-native parameters should be forwarded differently than for Gemini
   - Detect backend model type and adjust translation accordingly

---

### Phase 11: Testing & Verification (QUALITY)

**Sub-agent**: `testing-agent`

**Files to modify**:
- `tests/providers/antigravity/` (new directory)

**Tasks**:

1. **Create comprehensive translator tests**:
   - Test system prompt preservation with complex XML/markdown structures
   - Test tool schema sanitization preserves semantic fields
   - Test conversation structure doesn't inject unwanted messages
   - Test thinking block handling across turns
   - Test signature cache LRU behavior
   - Test SSE stream translation integrity

2. **Create integration smoke tests**:
   - Test round-trip: Anthropic request → Antigravity → Anthropic SSE
   - Test round-trip: Responses request → Antigravity → Responses SSE
   - Verify tool calling round-trip fidelity
   - Verify thinking/reasoning output fidelity

3. **Create regression tests for known hallucination patterns**:
   - Test that "Hello" and "Please continue." are not injected
   - Test that tool calls are never converted to text
   - Test that system prompt structure is preserved

---

### Phase 12: Performance & Reliability (POLISH)

**Sub-agent**: `performance-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/provider.py`
- `src/free_claude_code/providers/antigravity/auth.py`

**Tasks**:

1. **Add connection pooling optimization**:
   - Configure `httpx.AsyncClient` with proper connection limits and keep-alive
   - Add DNS caching for Antigravity hosts
   - Add HTTP/2 support for multiplexed streams

2. **Add proactive token refresh**:
   - Refresh tokens 10 minutes before expiry instead of 5
   - Add background refresh task that doesn't block requests

3. **Add request deduplication**:
   - For identical concurrent requests (same messages + model), deduplicate at the proxy level
   - Especially important for quota check and title generation requests

4. **Add response caching for deterministic requests**:
   - Cache responses for known-deterministic patterns (quota checks, title generation, filepath extraction)
   - Use short TTL (30s) to prevent stale results

---


---

### Phase 13: Multimodal & Document Fidelity (CRITICAL — Anti-Hallucination)

**Sub-agent**: `multimodal-fidelity-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Add PDF/document translation**: Map Anthropic `type="document"` to Gemini `inlineData` (with `application/pdf`).
2. **Fix tool result images**: When `btype == "tool_result"` contains images, map them to Gemini `inlineData` instead of dropping them.
3. **Preserve tool `is_error=True`**: When Anthropic tool results indicate an error, format the text string to explicitly signal an error state to Gemini.

---

### Phase 14: Context Caching Translation (CRITICAL — Performance)

**Sub-agent**: `caching-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Detect `cache_control`**: Find `cache_control: {"type": "ephemeral"}` in Anthropic `system` and `messages`.
2. **Translate to Gemini's `cachedContent` API**: Map ephemeral markers to Gemini's prompt caching mechanism.
3. **Emit Cache Tokens**: Ensure the proxy correctly signals cache hits/misses back to the client via `cache_creation_input_tokens` and `cache_read_input_tokens` in the final `usage` block.

---

### Phase 15: Schema & Token Accuracy (SIGNIFICANT — Reliability)

**Sub-agent**: `schema-token-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Fix missing required properties**: In `_sanitize_schema()`, if a property is in `required` but missing from `properties`, inject a generic `{"type": "ANY"}` property definition instead of dropping it from the `required` list.
2. **Fix input token tracking**: Read `promptTokenCount` from `usageMetadata` in `process_line()`.
3. **Emit accurate tokens**: Write accurate input and output token counts into `message_start` and `message_delta` events.


---

### Phase 16: Security & Resilience (SIGNIFICANT — Stability)

**Sub-agent**: `security-agent`

**Files to modify**:
- `src/free_claude_code/providers/antigravity/translator.py`

**Tasks**:

1. **Fix Unbounded Stream Buffer**: Add a maximum length constraint to `self._json_buffer` in `process_line()` (e.g., 5MB). If the buffer exceeds this size without forming valid JSON, raise a specific streaming error and clear the buffer rather than failing with OOM.
2. **Hardened JSON parsing**: Use safe JSON limits or chunked parsing for extremely large tool arguments to prevent CPU blocking on the main event loop during `json.loads`.

## 🎯 Priority Execution Order

Execute phases in this order for maximum impact:

```
Phase 1 (System Prompt)     ─┐
Phase 2 (Reasoning Budget)   ├─ CRITICAL: Do these first, in parallel
Phase 3 (Tool Calling)       │
Phase 4 (Conversation Struct)─┘

Phase 5 (Generation Config)  ─┐
Phase 6 (Stream Robustness)   ├─ SIGNIFICANT: Do second, in parallel
Phase 7 (Signature Cache)     │
Phase 8 (Responses API)      ─┘

Phase 9 (Error Handling)     ─┐
Phase 10 (Model Optimization) ├─ QUALITY: Do third, in parallel
Phase 11 (Testing)            │
Phase 12 (Performance)       ─┘

Phase 13 (Multimodal)        ─┐
Phase 14 (Context Caching)    ├─ CRITICAL: Execute immediately after Phase 1-4
Phase 15 (Schema/Tokens)     ─┘

Phase 16 (Security/OOM)      ── SIGNIFICANT: Prevent stream crashes
```

---

## 🤖 Sub-Agent Instructions

### For each phase's sub-agent:

1. **Read the full translator.py** (`src/free_claude_code/providers/antigravity/translator.py`) before making changes — it's the critical file
2. **Read the full provider.py** (`src/free_claude_code/providers/antigravity/provider.py`) for the streaming pipeline
3. **Read the models.py** (`src/free_claude_code/providers/antigravity/models.py`) for model aliases and capabilities
4. **Never break the existing API contract** — the proxy must continue to serve Anthropic Messages and OpenAI Responses format
5. **Test every change** by running `cd ~/Downloads/free-claude-code && uv run pytest tests/ -x -q` (if tests exist for your changes)
6. **Use type hints** and follow the existing code style (ruff formatted, Python 3.14)
7. **Add logging** with `loguru` for all new error paths
8. **Preserve all existing comments** unless they're wrong

### Key architectural constraints:
- The proxy translates Anthropic Messages API → Antigravity `streamGenerateContent` → Anthropic SSE events
- The proxy also translates OpenAI Responses API → (first to Anthropic Messages, then through the above pipeline)
- `AntigravityStreamTranslator` converts Gemini SSE chunks to Anthropic SSE events
- `translate_messages_request()` converts Anthropic Messages request to Antigravity payload
- The signature cache is critical for Gemini's `thoughtSignature` validation
- The `ProviderAdmissionController` manages rate limiting and retry logic

### File dependency graph:
```
routes.py → handlers/messages.py → execution.py → provider.py → translator.py → models.py
                                                                   ↑               ↑
                                                                   auth.py         ↑
                                                                                   ↑
routes.py → handlers/responses.py → execution.py → provider.py → translator.py ───┘
```

---

## ✅ Success Criteria

After all phases are complete, the proxy should:

1. **Preserve 100% of system prompt structure** — no structural degradation
2. **Allow full thinking budgets** — models can reason as deeply as they can natively
3. **Never convert tool calls to text** — always use structured function calling
4. **Never inject synthetic messages** — no "Hello" or "Please continue."
5. **Forward all generation parameters** — temperature, top_p, top_k, stop_sequences
6. **Handle SSE streams without corruption** — no truncated JSON, no split multi-byte characters
7. **Properly cache and validate signatures** — no 400 errors from mismatched signatures
8. **Recover gracefully from errors** — retry with backoff, not crash
9. **Support model-specific optimization** — different defaults for Gemini vs Claude
10. **Pass all regression tests** — no known hallucination patterns
11. **Preserve Document Modality** — no dropped PDFs
12. **Enable Context Caching** — huge system prompts are cached correctly
13. **Accurate Token Reporting** — UI displays true token usage

---

## 📝 Notes for the Human

- After the sub-agents complete their work, you can test the substitute proxy by running it from this directory
- If satisfied, you can replace the running version at your discretion
- The most impactful changes are in Phases 1-4 (system prompt, reasoning, tool calling, conversation structure)
- Phase 11 (testing) should be run after each phase to verify no regressions

