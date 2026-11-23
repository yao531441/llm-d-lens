---
name: auth
description: Use when adding, changing or deleting authenticated routes, permissions, roles, scopes, ownership, resource sharing, sessions or login/session behavior in the Python backend, Node gateway or browser UI.
---

# Auth (authentication & authorization)

Follow [AGENTS.md](../../../AGENTS.md) for discovery, verification and
unresolved decisions. The authoritative design is
[`docs/design/auth-rbac-design.md`](../../../docs/design/auth-rbac-design.md);
decisions are numbered there (D1–D26) and the implementation status is in §21/§22.
All paths below are repo-relative.

## When to use

Any change that touches who may do what: a new/renamed/removed API route, a new
resource type, a new page or action button, role/permission changes, ownership or
sharing, session/login/CSRF, or an internal Node→Python call. Combine with
[backend](../backend/SKILL.md) for the service/router shape, [database](../database/SKILL.md)
for owner/binding schema, and [ui](../ui/SKILL.md) for pages, nav and gates.

## The model in one paragraph

Authorization is **action × scope**. The **action** is a permission code
`domain:resource:action` the caller's role must contain; the **scope** is *which
instances* (global / cluster / one resource) the binding, ownership or an explicit
share covers. Both gates must pass. Python is the authoritative policy point; the
Node gateway is the primary browser enforcement point; the UI only hides what the
user cannot do (never enforce in the UI alone).

Implementation map:

- Permission catalog + built-in roles: `apps/backend/src/llm_d_lens/auth/permissions.py`
  (`ALL_PERMISSIONS`, `BUILTIN_ROLE_PERMISSIONS`, `BUILTIN_ROLE_SELF_SCOPED`).
  Codes are **code, not a table** — never add a capability only to the DB.
- Decision core: `apps/backend/src/llm_d_lens/auth/policy.py` (`authorize`), scope helpers in
  `apps/backend/src/llm_d_lens/auth/scope.py`.
- Request helpers: `apps/backend/src/llm_d_lens/auth/access.py`
  (`current_principal`, `visible_cluster_ids`, `filter_by_cluster`,
  `require_cluster_access`, `owner_for_create`, `resource_readable`).
- Route registry + default-deny middleware: `apps/backend/src/llm_d_lens/auth/routes.py`,
  `apps/backend/src/llm_d_lens/auth/context.py` (CSRF double-submit on unsafe methods).
- Browser entry and internal tools: `apps/backend/src/llm_d_lens/api/web.py`,
  `apps/backend/src/llm_d_lens/assistant_tools/tools.py`.
- Browser: `apps/frontend/src/api/httpClient.js`, `apps/frontend/src/features/auth/`,
  `apps/frontend/src/components/LeftNavigation.jsx`.

## Adding or changing a route (Python)

1. Put the handler in the owning domain router; the router-level permission comes
   from the registry, not from an inline `Depends`.
2. **Register the route** in `AUTH_ROUTE_PERMISSIONS` (`auth/routes.py`):
   `permission=None` = public, `""` = authenticated-only, otherwise a code in
   `ALL_PERMISSIONS`. Every non-whitelist route must have an entry — startup
   self-check fails fast otherwise. Missing the entry makes the route
   default-deny (`403 "route is not registered"`).
3. Compatibility, MCP and Playground routes use the same Python registry;
   preserve their existing permissions when moving handlers.
4. Enforce **resource scope**, not just the action: pass `current_principal(request)`
   and use `filter_by_cluster` (lists), `require_cluster_access` (target cluster),
   or `resource_readable(...)` (a concrete resource). A registered action alone
   only proves "may do this class of thing", not "may do it to this instance".
5. Server must derive identity/scope (cluster id, owner) from the record, never
   trust `clusterId`-style fields in the request body.
6. Add/adjust tests in `apps/backend/src/llm_d_lens/auth/` and the owning domain, and run the
   app import (startup self-check) plus `make test-python`.
7. **Removing a route or feature**: delete its registry entry, its `VIEW_PERMISSIONS`/nav/page
   entries, and revoke any resource shares it owned. Unused permission codes may
   stay in the catalog, but a removed built-in-role capability is a human
   decision.

## Adding a resource type (ownership + sharing)

A new shareable/owned resource needs, in the same task:

1. Owner columns (`owner_user_id` / `owner_group_id`, `SET NULL`) via a reviewed
   migration ([database](../database/SKILL.md)).
2. Record the owner on create with `owner_for_create(current_principal(request))`
   (thread it through background work too — see `start_run(owner_user_id=...)`).
3. Read paths call `resource_readable(...)` with the resource's `owner_*` and the
   right permission; list paths use `filter_by_cluster` / `visible_cluster_ids`.
4. Visibility rule (design §7.6): `visible within scope ∪ created by self ∪ owner-group membership ∪ explicitly shared`.
5. If end-user actions on it are self-scoped, add the action to
   `_END_USER_SELF_SCOPED`; otherwise a cluster binding grants it on everyone's
   rows.
6. If it can be shared, add the resource type to `SHARE_RESOURCE_TYPES`
   (`auth/service.py`) and use the existing share endpoints/service
   (`share_resource` / `revoke_resource_shares` / `list_resource_shares`).
7. **Purge shares on delete**: call `AuthService.revoke_resource_shares(...)` for
   the removed ids (executions/runs) so a stale share cannot outlive the resource
   or hit a reused id. Cluster deletion already cascades via `scope_cluster_id`.

## Self-scoped and creator-only rules (do not regress)

- An action in `BUILTIN_ROLE_SELF_SCOPED` is **not** granted by a cluster binding;
  it needs ownership or an explicit resource share.
- `OWNER_GROUP_EXCLUDED_ACTIONS` (`delete`, `share`, `grant-access`) are
  creator/maintainer-only: owner-group members and **resource shares** must not
  confer them.
- Casbin matcher uses `keyMatch`, **never `keyMatch2`** — codes are
  colon-delimited and keyMatch2 treats `:` as a wildcard, which leaks one
  resource's permission onto another.

## Backend entry and internal calls

- The Python auth middleware resolves browser sessions and validates
  `x-prism-internal-*` signatures for the exact method and path. Never trust
  unsigned client-supplied identity headers.
- Browser requests carry cookie + `x-prism-csrf` directly to the Python API.
- MCP dispatch in `apps/backend/src/llm_d_lens/assistant_tools/tools.py` signs internal ASGI
  requests for the current principal. Keep target route and resource permissions
  enforced even when the outer `/api/assistant/mcp` route accepts either `mcp:tool:invoke`
  or `playground:chat:use`. Playground uses the same tool dispatcher.
- Never add a browser-reachable path to the public whitelist to make an internal
  call work; fix the call's auth instead.

## Browser / page rules

1. **Mutations must use the shared transport** (`requestJson` / `postJson`) so
   credentials and the CSRF header are sent. Never use a bare `fetch` with a
   non-GET method — the backend rejects it with `403 csrf_failed`.
2. Gate views in `VIEW_PERMISSIONS` (`apps/frontend/src/features/auth/permissions.js`) and add
   the page to `SUPPORTED_VIEWS` (`apps/frontend/src/App.jsx`) and the nav
   (`apps/frontend/src/components/LeftNavigation.jsx`).
3. Gate actions with `PermissionGate` / `useAuth().can(...)`. UI gating is UX
   only — the backend still enforces.
4. When a page needs a picker/list that an end-user may not read via the admin
   APIs (users/groups/roles), add a purpose-limited, resource-scoped endpoint
   (see `.../access/options`) instead of opening the admin list.
5. Tag genuinely incomplete features with the existing `Exp` badge conventions
   (`LeftNavigation` item `badge`, `ModuleHeader badge`).

## Errors, CSRF and sessions

- Errors are RFC-7807 `application/problem+json` with a `code`. Unhandled
  exceptions now return `500` with `detail` (`<Type>: <message>`) and a
  `requestId` tied to the server traceback; the UI shows both. Never put secrets
  in exception messages.
- The Lens Assistant (`apps/backend/src/llm_d_lens/playground/chat.py`) treats authorization errors as
  final: its system prompt tells the model to stop (not retry or try other tools)
  and tell the user which permission is missing. Keep that rule in sync when you
  change 401/403 problem codes, and prefer a deterministic stop in the orchestrator
  when you add new permission-denied paths.
- Unsafe methods require the double-submit CSRF header; do not weaken
  `context._verify_csrf`.
- `password_change_required` blocks everything except the password/session/logout
  paths; keep `_PASSWORD_CHANGE_ALLOWED` in sync if you add such a route.

## Human decisions — stop and ask

Pause and get explicit approval (and record it as a new `D<n>` in the design doc
§18) before:

- changing a built-in role's permissions or the meaning of a permission code;
- adding/removing a public route or an authentication-mode bypass;
- any schema change (owner columns, binding tables) — [database](../database/SKILL.md);
- changing share/grant semantics or who may grant privileged roles;
- exposing new error detail or internal identity material.

## Verify

- Run auth and affected owning-domain tests directly. Run affected JavaScript
  tests and type checks when their contracts change. Shared permission or session
  changes require broader caller coverage; catalog scans do not replace tests.
- Restart the backend (`hack/lens-local.sh restart`) before manual verification:
  middleware, route registry and Node env changes are not hot-reloaded.
