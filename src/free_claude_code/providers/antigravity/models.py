"""Antigravity live model discovery, alias resolution, and capability mapping."""

from typing import Any

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality

DEFAULT_MODELS_HOSTS = (
    "daily-cloudcode-pa.googleapis.com",
    "cloudcode-pa.googleapis.com",
)

# Canonical aliases mapping user-facing model names to upstream backend model IDs
MODEL_ALIASES: dict[str, str] = {
    # Gemini models
    "gemini-3.8-flash": "gemini-3.8-flash-tiered",
    "gemini-3.8-flash-high": "gemini-3.8-flash-tiered",
    "gemini-3.7-flash": "gemini-3.7-flash-tiered",
    "gemini-3.7-flash-high": "gemini-3.7-flash-tiered",
    "gemini-3.6-flash": "gemini-3.6-flash-high",
    "gemini-3.6-flash-tiered": "gemini-3.6-flash-tiered",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    # Anthropic models supported on Antigravity
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
    "claude-opus-4-6": "claude-opus-4-6-thinking",
    "claude-opus-4.6": "claude-opus-4-6-thinking",
    "claude-opus": "claude-opus-4-6-thinking",
    # Other models
    "gpt-oss-120b-medium": "gpt-oss-120b-medium",
    # Claude 3.7 / 3.5 / 3 aliases (Claude Code CLI and Codex)
    "claude-3-7-sonnet": "claude-sonnet-4-6",
    "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-7-sonnet-20250219": "claude-sonnet-4-6",
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
    "claude-3-opus": "claude-opus-4-6-thinking",
    "claude-3-opus-latest": "claude-opus-4-6-thinking",
    "claude-3-opus-20240229": "claude-opus-4-6-thinking",
    "claude-3-5-haiku": "gemini-2.5-flash",
    "claude-3-5-haiku-latest": "gemini-2.5-flash",
    "claude-3-5-haiku-20241022": "gemini-2.5-flash",
    "claude-3-haiku": "gemini-2.5-flash",
    "claude-3-haiku-20240307": "gemini-2.5-flash",
    # OpenAI model aliases (Codex)
    "gpt-4o": "gemini-3.8-flash-tiered",
    "gpt-4o-mini": "gemini-2.5-flash",
    "o3-mini": "gemini-3.8-flash-tiered",
    "o1": "claude-opus-4-6-thinking",
    "o1-mini": "gemini-2.5-flash",
}

# Default known working Antigravity models (used if offline before discovery)
DEFAULT_ANTIGRAVITY_MODELS: tuple[ProviderModelInfo, ...] = (
    ProviderModelInfo(
        model_id="gemini-3.8-flash",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-3.8-flash-tiered",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-3.7-flash",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-3.7-flash-tiered",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-3.6-flash-high",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-3.1-pro-high",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-2.5-pro",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gemini-2.5-flash",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="claude-sonnet-4-6",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=200_000,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="claude-opus-4-6-thinking",
        supports_thinking=True,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE}),
        context_window_tokens=200_000,
        max_output_tokens=65_536,
    ),
    ProviderModelInfo(
        model_id="gpt-oss-120b-medium",
        supports_thinking=False,
        input_modalities=frozenset({ModelInputModality.TEXT}),
        context_window_tokens=128_000,
        max_output_tokens=16_384,
    ),
)


def resolve_backend_model(model_name: str) -> str:
    """Resolve a user-facing model name to the backend model ID expected by Antigravity."""
    clean = model_name.strip()
    if clean.startswith("antigravity/"):
        clean = clean[len("antigravity/") :]

    if clean.startswith("claude-google-"):
        suffix = clean[len("claude-google-") :]
        clean = suffix if suffix.startswith("gemini-") else f"gemini-{suffix}"

    # Check explicit alias table
    if clean in MODEL_ALIASES:
        return MODEL_ALIASES[clean]

    # Prefix matches for Claude models
    if clean.startswith("claude-3-7-sonnet") or clean.startswith("claude-3-5-sonnet") or clean.startswith("claude-sonnet"):
        return "claude-sonnet-4-6"
    if clean.startswith("claude-3-opus") or clean.startswith("claude-opus"):
        return "claude-opus-4-6-thinking"
    if clean.startswith("claude-3-5-haiku") or clean.startswith("claude-3-haiku") or clean.startswith("claude-haiku"):
        return "gemini-2.5-flash"

    # If user provided a name ending in clean alias (e.g. gemini-3.8-flash), check if -tiered exists
    tiered_candidate = f"{clean}-tiered"
    if tiered_candidate in MODEL_ALIASES.values():
        return tiered_candidate

    return clean


def parse_antigravity_models(payload: dict[str, Any]) -> frozenset[ProviderModelInfo]:
    """Parse live models returned by fetchAvailableModels into ProviderModelInfo set."""
    models_dict = payload.get("models")
    if not isinstance(models_dict, dict):
        return frozenset(DEFAULT_ANTIGRAVITY_MODELS)

    result: list[ProviderModelInfo] = []
    seen_ids: set[str] = set()

    for raw_id, info in models_dict.items():
        if not isinstance(raw_id, str) or not isinstance(info, dict):
            continue

        # Ignore internal experimental debug IDs without display names
        if (
            raw_id.startswith("chat_")
            or raw_id.startswith("tab_jump_")
            or raw_id.startswith("tab_")
        ):
            continue

        supports_thinking = bool(info.get("supportsThinking", False))
        supports_images = bool(info.get("supportsImages", False))
        modalities = {ModelInputModality.TEXT}
        if supports_images:
            modalities.add(ModelInputModality.IMAGE)

        max_tokens = info.get("maxTokens")
        max_output_tokens = info.get("maxOutputTokens")

        context_window = (
            int(max_tokens)
            if isinstance(max_tokens, int) and max_tokens > 0
            else 1_048_576
        )
        max_output = (
            int(max_output_tokens)
            if isinstance(max_output_tokens, int) and max_output_tokens > 0
            else 65_536
        )

        # Add the exact upstream model ID
        if raw_id not in seen_ids:
            seen_ids.add(raw_id)
            result.append(
                ProviderModelInfo(
                    model_id=raw_id,
                    supports_thinking=supports_thinking,
                    input_modalities=frozenset(modalities),
                    context_window_tokens=context_window,
                    max_output_tokens=max_output,
                )
            )

        # If model ends with -tiered, also expose the clean alias (e.g. gemini-3.8-flash)
        if raw_id.endswith("-tiered"):
            clean_alias = raw_id[: -len("-tiered")]
            if clean_alias not in seen_ids:
                seen_ids.add(clean_alias)
                result.append(
                    ProviderModelInfo(
                        model_id=clean_alias,
                        supports_thinking=supports_thinking,
                        input_modalities=frozenset(modalities),
                        context_window_tokens=context_window,
                        max_output_tokens=max_output,
                    )
                )

        # If model ends with -high, also expose the base alias (e.g. gemini-3.6-flash)
        if raw_id.endswith("-high"):
            clean_alias = raw_id[: -len("-high")]
            if clean_alias not in seen_ids:
                seen_ids.add(clean_alias)
                result.append(
                    ProviderModelInfo(
                        model_id=clean_alias,
                        supports_thinking=supports_thinking,
                        input_modalities=frozenset(modalities),
                        context_window_tokens=context_window,
                        max_output_tokens=max_output,
                    )
                )

    # Ensure popular aliases are present
    for alias, target in MODEL_ALIASES.items():
        if alias not in seen_ids and target in seen_ids:
            seen_ids.add(alias)
            # Find target info
            target_info = next((m for m in result if m.model_id == target), None)
            if target_info:
                result.append(
                    ProviderModelInfo(
                        model_id=alias,
                        supports_thinking=target_info.supports_thinking,
                        input_modalities=target_info.input_modalities,
                        context_window_tokens=target_info.context_window_tokens,
                        max_output_tokens=target_info.max_output_tokens,
                    )
                )

    return frozenset(result) if result else frozenset(DEFAULT_ANTIGRAVITY_MODELS)
