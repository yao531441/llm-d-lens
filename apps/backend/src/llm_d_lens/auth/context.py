"""FastAPI request-context middleware and route-permission dependency.

Node is the primary enforcement point; this is the Python-side defense in
depth (design sections 8.2/8.3). Middleware only resolves a principal (never
raises); the dependency makes the allow/deny decision for the matched route.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from llm_d_lens.auth.contracts import Principal, RoleBinding, ScopeType
from llm_d_lens.auth.permissions import ALL_PERMISSIONS
from llm_d_lens.auth.policy import authorize
from llm_d_lens.auth.routes import resolve_route_permission
from llm_d_lens.auth.security import verify_internal
from llm_d_lens.auth.service import default_service
from llm_d_lens.auth.settings import get_settings
from llm_d_lens.core.exceptions import (
    DomainError,
    ForbiddenError,
    PasswordChangeRequiredError,
    UnauthenticatedError,
)

SESSION_COOKIE = "prism_session"
CSRF_COOKIE = "prism_csrf"
CSRF_HEADER = "x-prism-csrf"


@dataclass
class AuditContext:
    request_id: str
    trace_id: str | None
    actor_user_id: str | None
    actor_username: str
    source: str
    method: str | None
    path: str | None
    ip: str | None
    user_agent: str | None
    permission: str | None = None
    parent_request_id: str | None = None
    event_type: str | None = None
    action: str | None = None
    cluster_id: str | None = None
    summary: str | None = None


_audit_context: contextvars.ContextVar[AuditContext | None] = contextvars.ContextVar("lens_audit_context", default=None)


def current_audit_context() -> AuditContext | None:
    return _audit_context.get()


def set_audit_event(event_type: str, *, action: str | None = None) -> None:
    context = current_audit_context()
    if context is not None:
        context.event_type = event_type
        if action is not None:
            context.action = action


def set_audit_cluster(cluster_id: str | None) -> None:
    context = current_audit_context()
    if context is not None:
        context.cluster_id = cluster_id


def set_audit_summary(summary: str | None) -> None:
    context = current_audit_context()
    if context is not None:
        context.summary = summary


def _trace_id(request: Request) -> str | None:
    header = request.headers.get("traceparent") or ""
    parts = header.split("-")
    if len(parts) >= 3 and len(parts[1]) == 32:
        return parts[1]
    return request.headers.get("x-trace-id") or None


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or request.headers.get("x-prism-request-id") or uuid.uuid4().hex


def _source(request: Request) -> str:
    has_session = bool(request.cookies.get(SESSION_COOKIE))
    has_origin = bool(request.headers.get("origin") or request.headers.get("referer"))
    if has_session or has_origin:
        return "browser"
    declared = request.headers.get("x-prism-source")
    if declared in {"assistant", "background"} and request.headers.get("x-prism-principal-id"):
        return declared
    if request.headers.get("x-prism-principal-id"):
        return "assistant" if request.url.path in {"/api/mcp", "/api/mcp/v2"} else "api"
    return "api"


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or None
    return request.client.host if request.client else None


class CsrfFailedError(DomainError):
    code = "csrf_failed"
    status_code = 403


def _verify_csrf(request: Request) -> None:
    """Double-submit cookie check (design section 12). Cookie-less clients
    (e.g. Bearer API callers) are exempt; browser sessions always carry it."""
    cookie = request.cookies.get(CSRF_COOKIE)
    if not cookie:
        return
    if request.headers.get(CSRF_HEADER) != cookie:
        raise CsrfFailedError("CSRF token is missing or does not match")


_PASSWORD_CHANGE_ALLOWED = {
    "/api/v1/auth/password",
    "/api/v1/auth/session",
    "/api/v1/auth/logout",
}


def token_from_request(request: Request) -> str | None:
    """Extract the opaque session token from the cookie or Bearer header."""
    cookie = request.cookies.get(SESSION_COOKIE)
    if cookie:
        return cookie
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def _disabled_principal() -> Principal:
    return Principal(
        user_id="system",
        username="system",
        bindings=(RoleBinding(role="admin", scope_type=ScopeType.GLOBAL),),
        role_permissions={"admin": frozenset(ALL_PERMISSIONS)},
        role_self_scoped={"admin": frozenset()},
        authz_version=0,
    )


def _internal_principal(request: Request) -> Principal | None:
    principal_id = request.headers.get("x-prism-principal-id")
    timestamp = request.headers.get("x-prism-internal-ts")
    signature = request.headers.get("x-prism-internal-sig")
    if not (principal_id and timestamp and signature):
        return None
    secret = get_settings().internal_auth_secret
    if not secret:
        return None
    try:
        timestamp_value = int(timestamp)
    except ValueError:
        return None
    if not verify_internal(
        secret,
        signature=signature,
        timestamp=timestamp_value,
        method=request.method,
        path=request.url.path,
        principal_id=principal_id,
    ):
        return None
    user = default_service().user_dao.get(principal_id)
    if user is None or user.status != "active":
        return None
    return default_service().principal_for(user)


class AuthContextMiddleware(BaseHTTPMiddleware):
    """Attach the resolved principal (or ``None``) to ``request.state``."""

    async def dispatch(self, request: Request, call_next):
        principal: Principal | None = None
        if get_settings().disabled:
            principal = _disabled_principal()
        else:
            principal = _internal_principal(request)
            if principal is None:
                token = token_from_request(request)
                if token:
                    principal = default_service().validate(token)
        request.state.principal = principal
        request.state.principal_token = token_from_request(request)
        found, permission = resolve_route_permission(request.method, request.url.path)
        context = AuditContext(
            request_id=_request_id(request),
            trace_id=_trace_id(request),
            actor_user_id=principal.user_id if principal and principal.user_id != "system" else None,
            actor_username=principal.username if principal else "",
            source=_source(request),
            method=request.method,
            path=request.url.path,
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            permission=(" | ".join(permission) if isinstance(permission, tuple) else permission) if found else None,
            parent_request_id=request.headers.get("x-audit-parent-request-id"),
        )
        token = _audit_context.set(context)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = context.request_id
            return response
        finally:
            _audit_context.reset(token)


async def require_route_permission(request: Request) -> None:
    """Global dependency: default-deny, with the registry deciding the code."""
    if get_settings().disabled:
        return
    found, permission = resolve_route_permission(request.method, request.url.path)
    if not found:
        raise ForbiddenError("route is not registered for authorization")
    if permission is None:
        return
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        _verify_csrf(request)
    principal: Principal | None = getattr(request.state, "principal", None)
    if principal is None:
        raise UnauthenticatedError("authentication required")
    if principal.must_change_password and request.url.path not in _PASSWORD_CHANGE_ALLOWED:
        raise PasswordChangeRequiredError("password change required before continuing")
    if permission == "":
        return
    permissions = permission if isinstance(permission, tuple) else (permission,)
    if not any(authorize(principal, candidate).allowed for candidate in permissions):
        default_service().audit(
            "access_denied",
            result="deny",
            actor_username=principal.username,
            permission=" | ".join(permissions),
            method=request.method,
            path=request.url.path,
            ip=_client_ip(request),
        )
        raise ForbiddenError("permission denied")
