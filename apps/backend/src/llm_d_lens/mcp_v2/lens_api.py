"""Transport-agnostic access to the Lens REST API.

Tools only see :class:`LensApi`. ``AsgiLensApi`` calls the running app in-process on behalf of the
authenticated caller; ``HttpLensApi`` talks to a remote Lens with a session token, which lets the same
tool layer run as a standalone stdio server before it is deployed next to Lens.
"""

from __future__ import annotations

import asyncio
import http.cookiejar
import json
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

import httpx


class LensApiError(Exception):
    def __init__(self, status: int, detail: Any, method: str, path: str):
        super().__init__(f"{method} {path} -> {status}")
        self.status = status
        self.detail = detail
        self.method = method
        self.path = path

    def message(self) -> str:
        detail = self.detail
        if isinstance(detail, dict):
            detail = detail.get("detail") or detail.get("message") or detail.get("error") or detail
        return str(detail)[:500]

    def code(self) -> str | None:
        return self.detail.get("code") if isinstance(self.detail, dict) else None


class LensApi(Protocol):
    async def request(
        self, method: str, path: str, *, params: dict[str, Any] | None = None, body: Any = None, timeout: float = 60
    ) -> Any:
        """Call a logical path such as ``/deployments/executions``; raise :class:`LensApiError` on non-2xx."""

    async def principal_id(self) -> str: ...


# Routers that are mounted without the ``/api/v1`` prefix inside the backend app.
_UNVERSIONED = {"cluster", "guide-planning"}


def _query(params: dict[str, Any] | None) -> str:
    pairs = []
    for key, value in (params or {}).items():
        if value is None or value == "":
            continue
        for item in value if isinstance(value, list) else [value]:
            pairs.append((key, str(item).lower() if isinstance(item, bool) else item))
    return "?" + urlencode(pairs) if pairs else ""


def _result(method: str, path: str, status: int, text: str) -> Any:
    try:
        data = json.loads(text) if text else None
    except ValueError:
        data = text
    if not 200 <= status < 300:
        raise LensApiError(status, data, method, path)
    return data


class AsgiLensApi:
    """In-process calls that carry the caller's own credentials, like the existing MCP endpoint."""

    def __init__(self, request):
        self._request = request

    async def principal_id(self) -> str:
        principal = getattr(self._request.state, "principal", None)
        if principal is None:
            raise LensApiError(401, "no authenticated principal", "-", "-")
        return principal.user_id

    def _headers(self, method: str, path: str) -> dict[str, str]:
        from llm_d_lens.auth.security import sign_internal
        from llm_d_lens.auth.settings import get_settings

        request = self._request
        headers = {
            key: request.headers[key] for key in ("cookie", "authorization", "x-prism-csrf") if key in request.headers
        }
        headers.update(accept="application/json", **{"x-prism-source": "assistant"})
        principal = getattr(request.state, "principal", None)
        secret = get_settings().internal_auth_secret
        if principal and secret:
            timestamp = int(time.time())
            headers.update(
                {
                    "x-prism-principal-id": principal.user_id,
                    "x-prism-internal-ts": str(timestamp),
                    "x-prism-internal-sig": sign_internal(
                        secret,
                        timestamp=timestamp,
                        method=method,
                        path=urlsplit(path).path,
                        principal_id=principal.user_id,
                    ),
                }
            )
        if request.headers.get("x-request-id"):
            headers["x-audit-parent-request-id"] = request.headers["x-request-id"]
        return headers

    async def request(self, method, path, *, params=None, body=None, timeout=60):
        root = path.lstrip("/").split("/", 1)[0]
        full = (f"/api{path}" if root in _UNVERSIONED else f"/api/v1{path}") + _query(params)
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self._request.app, raise_app_exceptions=False),
                base_url=str(self._request.base_url),
                timeout=timeout,
            ) as client,
        ):
            response = await client.request(method, full, json=body, headers=self._headers(method, full))
        return _result(method, path, response.status_code, response.text)


class HttpLensApi:
    """Remote Lens over HTTPS using a login session token as a Bearer credential."""

    def __init__(self, base_url: str, username: str, password: str, *, verify_tls: bool = False):
        self._base = base_url.rstrip("/")
        self._username = username
        self._password = password
        context = ssl.create_default_context() if verify_tls else ssl._create_unverified_context()  # noqa: S323
        self._jar = http.cookiejar.CookieJar()
        self._login_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
            urllib.request.HTTPCookieProcessor(self._jar),
        )
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
        )
        self._token: str | None = None
        self._principal: str | None = None

    def _login(self) -> None:
        payload = json.dumps({"username": self._username, "password": self._password}).encode()
        request = urllib.request.Request(  # noqa: S310
            f"{self._base}/api/v1/auth/login", data=payload, headers={"Content-Type": "application/json"}
        )
        self._login_opener.open(request, timeout=30).read()
        self._token = next((c.value for c in self._jar if c.name == "prism_session"), None)
        if not self._token:
            raise LensApiError(401, "login did not return a session cookie", "POST", "/auth/login")

    def _send(self, method: str, path: str, params, body, timeout: float) -> tuple[int, str]:
        for attempt in (0, 1):
            if self._token is None:
                self._login()
            headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
            data = None
            if body is not None:
                data = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(  # noqa: S310
                f"{self._base}/api/v1{path}{_query(params)}", data=data, headers=headers, method=method
            )
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as error:
                if error.code == 401 and attempt == 0:
                    self._token = None
                    continue
                return error.code, error.read().decode()
        raise AssertionError("unreachable")

    async def request(self, method, path, *, params=None, body=None, timeout=60):
        status, text = await asyncio.to_thread(self._send, method, path, params, body, timeout)
        return _result(method, path, status, text)

    async def principal_id(self) -> str:
        if self._principal is None:
            session = await self.request("GET", "/auth/session")
            self._principal = session["userId"]
        return self._principal
