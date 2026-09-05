"""Google Antigravity browser authorization flow with PKCE."""

import asyncio
import base64
import hashlib
import html
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
from aiohttp import web

GOOGLE_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

ANTIGRAVITY_CLIENT_ID = (
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
)
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/aicode",
)
ANTIGRAVITY_CALLBACK_HOST = "localhost"
LOGIN_TIMEOUT_SECONDS = 15 * 60

# Candidate loopback ports to try for the redirect listener
_CANDIDATE_PORTS = (
    51111,
    51112,
    51113,
    51114,
    51115,
    51116,
    51117,
    51118,
    51119,
    51120,
)


class AntigravityLoginError(RuntimeError):
    """An interactive Antigravity authorization flow could not complete."""


@dataclass(frozen=True, slots=True, repr=False)
class AuthorizationGrant:
    """OAuth authorization code and matching PKCE material."""

    code: str
    redirect_uri: str
    code_verifier: str


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _callback_page(
    title: str, message: str, *, status: int = 200, close_window: bool = False
) -> web.Response:
    close_script = (
        "<script>setTimeout(function(){ window.close(); }, 1500);</script>"
        if close_window
        else ""
    )
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{safe_title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: grid;
      place-items: center;
      min-height: 100vh;
      margin: 0;
      background: #0f172a;
      color: #f8fafc;
    }}
    .card {{
      background: #1e293b;
      padding: 2.5rem;
      border-radius: 1rem;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
      max-width: 28rem;
      text-align: center;
      border: 1px solid #334155;
    }}
    h1 {{
      font-size: 1.5rem;
      margin-top: 0;
      margin-bottom: 0.75rem;
      color: #38bdf8;
    }}
    p {{
      color: #94a3b8;
      line-height: 1.5;
      margin: 0;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{safe_title}</h1>
    <p>{safe_message}</p>
  </div>
  {close_script}
</body>
</html>"""
    return web.Response(text=page, content_type="text/html", status=status)


class BrowserAuthorization:
    """Short-lived loopback callback server for Google Antigravity PKCE login."""

    def __init__(
        self,
        *,
        auth_url: str,
        redirect_uri: str,
        code_verifier: str,
        runner: web.AppRunner,
        result: asyncio.Future[AuthorizationGrant],
    ) -> None:
        self.auth_url = auth_url
        self.redirect_uri = redirect_uri
        self.code_verifier = code_verifier
        self._runner = runner
        self._result = result
        self._closed = False

    @classmethod
    async def start(cls) -> BrowserAuthorization:
        """Bind an available loopback port and construct the authorization URL."""
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
        loop = asyncio.get_running_loop()
        result: asyncio.Future[AuthorizationGrant] = loop.create_future()
        runner: web.AppRunner | None = None
        redirect_uri = ""

        # Try designated ports first, then port 0 (dynamic assigned port)
        candidate_ports = (*_CANDIDATE_PORTS, 0)
        for candidate in candidate_ports:
            app = web.Application()

            async def callback(request: web.Request) -> web.Response:
                if request.query.get("state") != state:
                    return web.Response(status=400, text="State mismatch")
                if error := request.query.get("error"):
                    description = request.query.get("error_description", error)
                    if not result.done():
                        result.set_exception(AntigravityLoginError(description))
                    return _callback_page("Sign-in failed", description, status=400)
                code = request.query.get("code")
                if not code:
                    return web.Response(status=400, text="Missing authorization code")
                if not result.done():
                    result.set_result(
                        AuthorizationGrant(
                            code=code,
                            redirect_uri=request.app.get("redirect_uri", ""),
                            code_verifier=verifier,
                        )
                    )
                return _callback_page(
                    "Connected to Google Antigravity",
                    "Google Antigravity is now connected to Free Claude Code. You can close this tab.",
                    close_window=True,
                )

            app.router.add_get("/auth/callback", callback)
            candidate_runner = web.AppRunner(app, access_log=None)
            await candidate_runner.setup()
            try:
                candidate_site = web.TCPSite(
                    candidate_runner,
                    ANTIGRAVITY_CALLBACK_HOST,
                    candidate,
                )
                await candidate_site.start()
            except OSError:
                await candidate_runner.cleanup()
                continue

            bound_port = candidate
            server = getattr(candidate_site, "_server", None)
            if server is not None and getattr(server, "sockets", None):
                sock = server.sockets[0]
                bound_port = sock.getsockname()[1]

            redirect_uri = (
                f"http://{ANTIGRAVITY_CALLBACK_HOST}:{bound_port}/auth/callback"
            )
            app["redirect_uri"] = redirect_uri
            runner = candidate_runner
            break

        if runner is None or not redirect_uri:
            raise AntigravityLoginError(
                "Could not bind a localhost callback port for Antigravity sign-in."
            )

        auth_url = f"{GOOGLE_OAUTH_AUTH_URL}?" + urlencode(
            {
                "response_type": "code",
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": " ".join(ANTIGRAVITY_SCOPES),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "access_type": "offline",
                "prompt": "consent",
            }
        )
        return cls(
            auth_url=auth_url,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
            runner=runner,
            result=result,
        )

    async def wait(self) -> AuthorizationGrant:
        return await self._result

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.cancel()
        await self._runner.cleanup()


async def exchange_authorization_code(
    client: httpx.AsyncClient,
    grant: AuthorizationGrant,
) -> dict[str, object]:
    """Exchange authorization code for tokens."""
    payload = {
        "grant_type": "authorization_code",
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "client_secret": ANTIGRAVITY_CLIENT_SECRET,
        "code": grant.code,
        "redirect_uri": grant.redirect_uri,
        "code_verifier": grant.code_verifier,
    }
    response = await client.post(
        GOOGLE_OAUTH_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if not response.is_success:
        detail = response.text
        try:
            error_data = response.json()
            detail = (
                error_data.get("error_description") or error_data.get("error") or detail
            )
        except Exception:
            pass
        raise AntigravityLoginError(f"Antigravity token exchange failed: {detail}")

    data = response.json()
    if not isinstance(data, dict):
        raise AntigravityLoginError("Malformed token response from Google OAuth.")
    return data
