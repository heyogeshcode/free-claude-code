"""Local admin UI routes and APIs."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.values import load_config_response, load_value_state
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.version import package_version

from .admin_security import require_loopback_admin
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
PACKAGE_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_ADMIN_ASSET_VERSION_PLACEHOLDER = "__FCC_VERSION__"
_ADMIN_ASSET_FILENAMES = frozenset(
    {
        "admin.css",
        "admin.js",
        "app-icon.svg",
        "chat_sessions.css",
        "chat_sessions.js",
        "model_combobox.js",
    }
)
LOCAL_PROVIDER_PATHS = {
    "lmstudio": "/models",
    "llamacpp": "/models",
    "ollama": "/api/tags",
}
_LOCAL_PROVIDER_CHECK_FAILURE_MESSAGE = (
    "Could not connect. Verify the URL and that the local provider is running."
)


class AdminConfigPayload(BaseModel):
    """Partial config update submitted by the admin UI."""

    values: JsonObject = Field(default_factory=dict)


class ConnectedAccountLoginPayload(BaseModel):
    """Interactive connected-account login selection."""

    mode: ConnectedAccountLoginMode | None = None


def _asset_path(filename: str) -> Path:
    asset_dir = PACKAGE_ASSETS_DIR if filename == "app-icon.svg" else STATIC_DIR
    path = asset_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return path


def _asset_response(filename: str) -> FileResponse:
    return FileResponse(_asset_path(filename))


def admin_page_response() -> HTMLResponse:
    template = _asset_path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        template.replace(_ADMIN_ASSET_VERSION_PLACEHOLDER, package_version())
    )


@router.get("/admin", include_in_schema=False)
def admin_page(request: Request):
    require_loopback_admin(request)
    return admin_page_response()


@router.get("/admin/assets/{version}/{filename}", include_in_schema=False)
async def admin_asset(version: str, filename: str, request: Request):
    require_loopback_admin(request)
    if version != package_version() or filename not in _ADMIN_ASSET_FILENAMES:
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return _asset_response(filename)


@router.get("/admin/api/config")
async def get_admin_config(request: Request):
    require_loopback_admin(request)
    return load_config_response()


@router.post("/admin/api/config/apply")
async def apply_admin_config(
    payload: AdminConfigPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.apply_admin_config(_filtered_values(payload.values))
    restart = result.get("restart")
    if isinstance(restart, dict) and restart.get("automatic"):
        background_tasks.add_task(services.admin.request_restart)
    return result


@router.get("/admin/api/status")
async def admin_status(
    request: Request,
    response: Response,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    # A local Admin page may reconnect after Apply changes the listening port.
    # The existing security check admits only loopback callers and origins.
    if origin := request.headers.get("origin"):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return services.admin.admin_status()


@router.get("/admin/api/providers/local-status")
async def local_provider_status(request: Request):
    require_loopback_admin(request)
    values = {key: entry.value or "" for key, entry in load_value_state().items()}
    checks = await asyncio.gather(
        *(
            _check_local_provider(
                provider_id,
                _local_provider_url(provider_id, values),
                path,
            )
            for provider_id, path in LOCAL_PROVIDER_PATHS.items()
        )
    )
    return {"providers": checks}


@router.post("/admin/api/providers/{provider_id}/test")
async def test_provider(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return await services.admin.test_provider(provider_id)


@router.get("/admin/api/providers/{provider_id}/auth")
async def connected_account_status(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.connected_account_status(provider_id)
    return _no_store(status.as_dict())


@router.post("/admin/api/providers/{provider_id}/auth/login")
async def start_connected_account_login(
    provider_id: str,
    payload: ConnectedAccountLoginPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    account = await services.admin.connected_account_status(provider_id)
    mode = payload.mode or account.default_login_mode
    if mode not in account.supported_login_modes:
        raise HTTPException(
            status_code=422,
            detail="Login mode is not supported by this provider.",
        )
    try:
        status = await services.admin.start_connected_account_login(provider_id, mode)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(f"Could not start connected-account login ({type(exc).__name__})."),
        ) from exc
    return _no_store(status.as_dict())


@router.post("/admin/api/providers/{provider_id}/auth/cancel")
async def cancel_connected_account_login(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.cancel_connected_account_login(provider_id)
    return _no_store(status.as_dict())


@router.delete("/admin/api/providers/{provider_id}/auth")
async def disconnect_connected_account(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.disconnect_connected_account(provider_id)
    return _no_store(status.as_dict())


def _resolve_account_email(acc: object) -> str:
    label = getattr(acc, "label", "") or ""
    if "@" in label:
        return label
    creds = getattr(acc, "credentials", {}) or {}
    if isinstance(creds, dict):
        if "email" in creds and creds["email"]:
            return str(creds["email"])
        inner_creds = creds.get("credentials")
        if isinstance(inner_creds, dict) and inner_creds.get("email"):
            return str(inner_creds["email"])
        id_token = creds.get("id_token")
        if not id_token and isinstance(inner_creds, dict):
            id_token = inner_creds.get("id_token")
        if isinstance(id_token, str) and "." in id_token:
            try:
                import base64
                import json

                parts = id_token.split(".")
                if len(parts) >= 2:
                    padding = "=" * (4 - len(parts[1]) % 4)
                    data = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
                    if "email" in data and data["email"]:
                        return str(data["email"])
                    profile = data.get("https://api.openai.com/profile")
                    if isinstance(profile, dict) and profile.get("email"):
                        return str(profile["email"])
            except Exception:
                pass
    return label or getattr(acc, "id", "")


def _serialize_accounts(provider_id: str) -> dict[str, object]:
    from free_claude_code.application.account_store import get_account_store

    store = get_account_store()
    accounts = store.get_accounts(provider_id)

    acc_pcts: list[float] = []
    active_count = 0
    cooling_count = 0
    serialized_accounts = []

    for acc in accounts:
        email = _resolve_account_email(acc)
        is_cooling = any(acc.is_cooling(m) for m in acc.model_cooldowns)
        if acc.is_active() and not is_cooling:
            active_count += 1
            acc_pct = 100.0
        elif acc.is_active() and is_cooling:
            cooling_count += 1
            max_rem = max(
                (acc.cooldown_remaining(m) for m in acc.model_cooldowns),
                default=0.0,
            )
            acc_pct = max(0.0, 100.0 - (max_rem / 60.0) * 100.0)
        else:
            acc_pct = 0.0
        acc_pcts.append(acc_pct)

        serialized_accounts.append(
            {
                "id": acc.id,
                "label": email,
                "raw_label": acc.label,
                "email": email,
                "priority": acc.priority,
                "status": acc.status,
                "is_cooling": is_cooling,
                "cooling_models": [
                    {
                        "model": m,
                        "remaining_seconds": round(acc.cooldown_remaining(m), 1),
                    }
                    for m in acc.model_cooldowns
                    if acc.is_cooling(m)
                ],
                "remaining_pct": round(acc_pct, 1),
            }
        )

    combined_pct = round(sum(acc_pcts) / len(acc_pcts)) if acc_pcts else 0

    return {
        "provider_id": provider_id,
        "accounts": serialized_accounts,
        "limit_summary": {
            "remaining_pct": combined_pct,
            "total_accounts": len(accounts),
            "active_accounts": active_count,
            "cooling_accounts": cooling_count,
        },
    }


@router.get("/admin/api/providers/{provider_id}/accounts")
async def list_provider_accounts(
    provider_id: str,
    request: Request,
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    return _no_store(_serialize_accounts(provider_id))


@router.post("/admin/api/providers/{provider_id}/accounts/{account_id}/move-up")
async def move_account_up(
    provider_id: str,
    account_id: str,
    request: Request,
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    from free_claude_code.application.account_store import get_account_store
    store = get_account_store()
    success = store.move_up(provider_id, account_id)
    return _no_store({"ok": success, **_serialize_accounts(provider_id)})


@router.post("/admin/api/providers/{provider_id}/accounts/{account_id}/move-down")
async def move_account_down(
    provider_id: str,
    account_id: str,
    request: Request,
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    from free_claude_code.application.account_store import get_account_store
    store = get_account_store()
    success = store.move_down(provider_id, account_id)
    return _no_store({"ok": success, **_serialize_accounts(provider_id)})


@router.delete("/admin/api/providers/{provider_id}/accounts/{account_id}")
async def delete_provider_account(
    provider_id: str,
    account_id: str,
    request: Request,
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    from free_claude_code.application.account_store import get_account_store
    store = get_account_store()
    success = store.delete_account(provider_id, account_id)
    return _no_store({"ok": success, **_serialize_accounts(provider_id)})


@router.post("/admin/api/providers/{provider_id}/accounts/{account_id}/clear-cooldown")
async def clear_account_cooldown(
    provider_id: str,
    account_id: str,
    request: Request,
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    from free_claude_code.application.account_store import get_account_store
    store = get_account_store()
    acc = store.get_account_by_id(provider_id, account_id)
    if acc:
        acc.model_cooldowns.clear()
        store.save()
    return _no_store({"ok": True, **_serialize_accounts(provider_id)})


@router.get("/admin/api/oauth/limits")
async def oauth_limits(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    from free_claude_code.application.account_store import get_account_store

    store = get_account_store()
    provider_names = {
        "antigravity": "Google Antigravity",
        "openai": "OpenAI / ChatGPT",
        "github_copilot": "GitHub Copilot",
    }

    results = []
    for provider_id in ("antigravity", "openai", "github_copilot"):
        try:
            acc_status = await services.admin.connected_account_status(provider_id)
            if not acc_status.connected:
                continue
        except Exception:
            continue

        data = _serialize_accounts(provider_id)
        accounts = data.get("accounts", [])
        if not accounts:
            continue

        summary = data.get("limit_summary", {})
        results.append(
            {
                "provider_id": provider_id,
                "display_name": provider_names.get(provider_id, provider_id),
                "remaining_pct": summary.get("remaining_pct", 100),
                "total_accounts": summary.get("total_accounts", len(accounts)),
                "active_accounts": summary.get("active_accounts", len(accounts)),
                "cooling_accounts": summary.get("cooling_accounts", 0),
                "accounts": accounts,
            }
        )

    return _no_store({"providers": results})


@router.get("/admin/api/models")
async def models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return _model_options(services)


@router.post("/admin/api/models/refresh")
async def refresh_models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.refresh_models()
    return _model_options(services, refresh_result=result)


def _model_options(
    services: ApiServices,
    *,
    refresh_result: ProviderModelRefreshResult | None = None,
) -> dict[str, list[str]]:
    configured = {
        ref.model_ref
        for ref in configured_chat_model_refs(services.requests.current_settings())
    }
    discovered = {
        info.model_id for info in services.requests.cached_prefixed_model_infos()
    }
    failed_provider_ids = (
        refresh_result.failed_provider_ids if refresh_result is not None else ()
    )
    return {
        "models": sorted(configured | discovered, key=str.casefold),
        "failed_providers": list(failed_provider_ids),
    }


def _filtered_values(values: Mapping[str, JsonValue]) -> JsonObject:
    return {key: value for key, value in values.items() if key in FIELD_BY_KEY}


def _local_provider_url(provider_id: str, values: dict[str, str]) -> str:
    if provider_id == "lmstudio":
        return values.get("LM_STUDIO_BASE_URL", "")
    if provider_id == "llamacpp":
        return values.get("LLAMACPP_BASE_URL", "")
    if provider_id == "ollama":
        return values.get("OLLAMA_BASE_URL", "")
    return ""


async def _check_local_provider(
    provider_id: str, base_url: str, path: str
) -> JsonObject:
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return {
            "provider_id": provider_id,
            "status": "missing_url",
            "label": "Missing URL",
            "base_url": base_url,
        }

    url = f"{clean_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(url)
        ok = 200 <= response.status_code < 300
        return {
            "provider_id": provider_id,
            "status": "reachable" if ok else "offline",
            "label": "Reachable" if ok else "Offline",
            "base_url": base_url,
            "status_code": response.status_code,
        }
    except Exception as exc:
        logger.debug(
            "Admin local provider check failed: provider={} exc_type={}",
            provider_id,
            type(exc).__name__,
        )
        return {
            "provider_id": provider_id,
            "status": "offline",
            "label": "Offline",
            "base_url": base_url,
            "message": _LOCAL_PROVIDER_CHECK_FAILURE_MESSAGE,
        }


def _require_connected_account_provider(provider_id: str) -> None:
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if (
        descriptor is None
        or descriptor.auth_kind is not ProviderAuthKind.CONNECTED_ACCOUNT
    ):
        raise HTTPException(
            status_code=404,
            detail="Provider does not support connected-account login.",
        )


def _no_store(payload: JsonValue) -> JSONResponse:
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})
