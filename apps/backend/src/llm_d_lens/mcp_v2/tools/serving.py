"""Serving profile: publish deployments as model services behind Lens's managed gateway and manage their traffic."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..context import ToolContext
from ..deployment_refs import resolve
from ..errors import ToolError
from ..lens_api import LensApiError
from ..registry import tool
from ..serving_gateway import TOKEN_ENV, gateway_token, send_chat
from ..serving_ops import (
    ADMIN,
    ServiceRef,
    conflict_error,
    connection_info,
    delete,
    gateway_base_url,
    is_definite_failure,
    lens_write,
    load_service,
    outcome_unknown,
    owned_deployment_ids,
    reconcile_service,
    require_owned,
    resolve_service,
    view_of,
)
from ..serving_view import is_routable, service_version

CREATED_VIA = "lens-mcp-v2"
SUBJECT_PATTERN = re.compile(r"^(user|group):[A-Za-z0-9._@-]+$")
TOKENS_IN_UI = "Service tokens are created in the Lens UI only; no tool creates or returns tokens."


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServiceIdArgs(Args):
    service_id: str = Field(
        description="Service id (svc-...) from publish_model_service or list_resources(kinds=['model_service'])."
    )


# --- publish ---------------------------------------------------------------------------------------------


class PublishArgs(Args):
    public_name: str = Field(
        pattern=r"^[a-z0-9]([a-z0-9-]{0,40}[a-z0-9])?$",
        description="Public model name clients put in the 'model' field. Unique per cluster.",
    )
    deployment_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Your own ready llm-d deployments (dep-...) serving the same model, "
            "from list_resources(kinds=['deployment'])."
        ),
    )
    weights: dict[str, int] | None = Field(
        default=None,
        description="Optional relative traffic weights keyed by deployment id (0-100 each). Default: equal.",
    )
    description: str = Field(default="", max_length=500, description="Human-readable description.")
    ttl_hours: int = Field(default=8, ge=1, le=72, description="Recorded lifetime in hours; nothing reaps it yet.")
    client_request_id: str | None = Field(default=None, description="Retry key: repeating it returns the first result.")


async def _publishable(ctx: ToolContext) -> dict[str, dict[str, Any]]:
    items = (await ctx.api.request("GET", f"{ADMIN}/deployments")).get("items") or []
    return {item["executionId"]: item for item in items}


async def _owned_executions(ctx: ToolContext, deployment_ids: list[str], field: str) -> dict[str, str]:
    executions = {}
    for deployment_id in deployment_ids:
        ref = await resolve(ctx, deployment_id)
        if not ref.owned:
            raise ToolError(
                "not_owner" if ref.foreign_owner else "owner_unknown",
                f"Deployment '{deployment_id}' was not created by you through this server.",
                field=field,
                hint="Only your own deployments can be published. Check list_resources(kinds=['deployment']).",
            )
        if not ref.execution_id:
            raise ToolError(
                "conflict_state",
                f"Deployment '{deployment_id}' has no running execution yet.",
                field=field,
                hint="Wait until get_deployment reports ready=true.",
            )
        executions[deployment_id] = ref.execution_id
    return executions


async def _require_free_name(ctx: ToolContext, name: str, cluster_id: str) -> None:
    groups = (await ctx.api.request("GET", f"{ADMIN}/groups")).get("items") or []
    for group in groups:
        if group.get("clusterId") == cluster_id and group["name"].lower() == name.lower():
            pending = next(
                (
                    r
                    for r in ctx.ledger.list_resources(owner=ctx.owner, kind="model_service", cluster_id=cluster_id)
                    if r.state == "creating" and not r.lens_execution_id and r.name == name
                ),
                None,
            )
            if pending is not None:
                confirmed = await reconcile_service(ctx, pending)
                if confirmed.lens_execution_id:
                    raise ToolError(
                        "already_exists",
                        f"Your earlier publish of '{name}' did complete: it is service {confirmed.id}.",
                        field="public_name",
                        hint=f"Use get_model_service(service_id='{confirmed.id}').",
                        service_id=confirmed.id,
                    )
            mine = ctx.ledger.find_by_execution(group["id"])
            whose = "one of your services" if mine and mine.owner == ctx.owner else "another service"
            raise ToolError(
                "already_exists",
                f"public_name '{name}' is already used by {whose} on this cluster.",
                field="public_name",
                hint="Pick another name, or change the existing service with update_model_service.",
            )


async def _put_traffic(
    ctx: ToolContext, group_id: str, expected: dict[str, int], desired: dict[str, int], warnings: list[str]
) -> None:
    body = {"members": [{"executionId": e, "weight": w} for e, w in desired.items()], "expectedWeights": expected}
    await lens_write(ctx, "PUT", f"{ADMIN}/groups/{group_id}/traffic", body, warnings)


@tool(
    "publish_model_service",
    "[W] Publish your own ready deployment(s) of one model as a stable OpenAI-compatible endpoint behind Lens's "
    "managed gateway (HTTPRoute). Only you can call it until you grant others with grant_model_service_access. "
    "Verify with test_model_service. Use update_model_service to change an existing service. "
    "No token is created or returned; service tokens come from the Lens UI.\n"
    "Side effects: creates a model-service record and a gateway route; resources: no extra accelerators; "
    "duration: seconds, routing may take a little longer; reversible: delete_resources(ids=[svc-...]).",
    PublishArgs,
    read_only=False,
    idempotent=True,
    open_world=True,
    profile="serving",
)
async def publish_model_service(ctx: ToolContext, args: PublishArgs) -> dict[str, Any]:
    executions = await _owned_executions(ctx, args.deployment_ids, "deployment_ids")
    weights = args.weights or {}
    unknown = sorted(set(weights) - set(executions))
    if unknown:
        raise ToolError("invalid_argument", f"weights names {unknown}, not in deployment_ids.", field="weights")
    if any(not 0 <= w <= 100 for w in weights.values()):
        raise ToolError("invalid_argument", "Weights must be between 0 and 100.", field="weights")
    publishable = await _publishable(ctx)
    chosen = []
    for deployment_id, execution_id in executions.items():
        item = publishable.get(execution_id)
        if item is None:
            raise ToolError(
                "conflict_state",
                f"Deployment '{deployment_id}' cannot be published right now.",
                field="deployment_ids",
                hint="It must be ready and an llm-d deployment with an InferencePool and EPP (plain vLLM is not "
                "supported). Check get_deployment.",
            )
        chosen.append(item)
    if len({item["clusterId"] for item in chosen}) > 1 or len({item["model"] for item in chosen}) > 1:
        raise ToolError(
            "unsupported_combination",
            "All members must serve the same model on the same cluster.",
            field="deployment_ids",
        )
    cluster_id = chosen[0]["clusterId"]
    shares = {execution: weights.get(dep, 1) for dep, execution in executions.items()}
    if sum(shares.values()) <= 0:
        raise ToolError("invalid_argument", "At least one weight must be above 0.", field="weights")
    await _require_free_name(ctx, args.public_name, cluster_id)

    row = ctx.ledger.add_resource(
        kind="model_service",
        owner=ctx.owner,
        cluster_id=cluster_id,
        name=args.public_name,
        state="creating",
        meta={
            "created_via": CREATED_VIA,
            "expires_at": time.time() + args.ttl_hours * 3600,
            "deployments": list(executions),
        },
    )
    warnings: list[str] = []
    first, *others = executions.values()
    try:
        group = await lens_write(
            ctx,
            "POST",
            f"{ADMIN}/groups/publish",
            {
                "name": args.public_name,
                "clusterId": cluster_id,
                "executionId": first,
                "displayName": args.public_name,
                "description": args.description,
            },
            warnings,
        )
    except asyncio.CancelledError:
        raise  # the request may have reached Lens; keep the row so get_model_service can reconcile it
    except Exception as error:
        if is_definite_failure(error):
            ctx.ledger.set_state(row.id, "deleted")
            raise
        raise outcome_unknown(row, error) from error
    if group is None:
        # Lens saved the group but could not apply routing yet: the service exists.
        confirmed = await reconcile_service(ctx, row)
        if not confirmed.lens_execution_id:
            raise outcome_unknown(row, RuntimeError("no response body"))
        group = {"id": confirmed.lens_execution_id}
    else:
        ctx.ledger.bind_execution(row.id, group["id"])
        ctx.ledger.set_state(row.id, "active")
    try:
        for execution_id in others:
            try:
                await ctx.api.request(
                    "POST", f"{ADMIN}/members", body={"groupId": group["id"], "executionId": execution_id}, timeout=120
                )
            except LensApiError as error:
                raise conflict_error(error) from error
        if others and set(shares.values()) != {1}:
            await _put_traffic(ctx, group["id"], dict.fromkeys(shares, 1), shares, warnings)
    except Exception:
        try:
            await delete(ctx, ctx.ledger.get(row.id), True)
            ctx.ledger.set_state(row.id, "deleted")
        except Exception:  # noqa: BLE001 - report instead of masking the original failure
            warnings.append(f"Partial publish could not be rolled back; delete {row.id} with delete_resources.")
        raise
    ref = ServiceRef(ctx.ledger.get(row.id), group["id"])
    group, members = await load_service(ctx, group["id"])
    view = await view_of(ctx, ref, group, members)
    return {
        **view,
        "access": "owner only",
        "warnings": warnings,
        "next": f"Call test_model_service(service_id='{row.id}'). {TOKENS_IN_UI}",
    }


# --- get -------------------------------------------------------------------------------------------------


class GetServiceArgs(ServiceIdArgs):
    include: list[Literal["access", "access_options", "usage", "connection"]] = Field(
        default_factory=list,
        description="access=grants and your token metadata (never token values); access_options=users and groups "
        "that can be granted; usage=request and token totals; connection=curl/OpenAI client snippets. "
        "access, access_options and usage are owner-only.",
    )
    usage_window_hours: int = Field(default=24, ge=1, le=720, description="Window for include=usage.")
    wait_seconds: int = Field(default=0, ge=0, le=25, description="Wait up to this long while status is 'publishing'.")


async def _access_section(ctx: ToolContext, group_id: str) -> dict[str, Any]:
    grants = (await ctx.api.request("GET", f"{ADMIN}/groups/{group_id}/access")).get("items") or []
    tokens = (await ctx.api.request("GET", "/model-service/tokens")).get("items") or []
    return {
        "grants": [
            {
                "subject": f"{g['subjectType']}:{g.get('subjectName') or g['subjectId']}",
                "binding_id": g["id"],
                "granted_at": g.get("createdAt"),
            }
            for g in grants
        ],
        "your_tokens": [
            {
                "token_id": t["id"],
                "name": t.get("name"),
                "created_at": t.get("createdAt"),
                "last_used_at": t.get("lastUsedAt"),
                "expires_at": t.get("expiresAt"),
            }
            for t in tokens
        ],
        "note": "Tokens are account-wide: a token calls every service its user may call.",
    }


async def _usage_section(ctx: ToolContext, group_id: str, hours: int) -> dict[str, Any]:
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    params = {"group_id": group_id, "since": since, "group_by": "model"}
    totals = (await ctx.api.request("GET", f"{ADMIN}/usage/analytics", params=params)).get("totals") or {}
    return {
        "window_hours": hours,
        "requests": totals.get("requests", 0),
        "input_tokens": totals.get("inputTokens", 0),
        "output_tokens": totals.get("outputTokens", 0),
    }


@tool(
    "get_model_service",
    "[R] Show a model service: endpoint (no token), status, members with weights and health, and a version for "
    "update_model_service(expected_version). Optional sections: access, access_options, usage, connection. "
    "Works for services of others in a read-only view. Use list_resources to find services.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    GetServiceArgs,
    read_only=True,
    profile="serving",
)
async def get_model_service(ctx: ToolContext, args: GetServiceArgs) -> dict[str, Any]:
    ref = await resolve_service(ctx, args.service_id)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        group, members = await load_service(ctx, ref.group_id)
        view = await view_of(ctx, ref, group, members)
        if view["status"] != "publishing" or time.monotonic() >= deadline:
            break
        await asyncio.sleep(2)
    if "connection" in args.include:
        view["connection"] = connection_info(view)
    sections: dict[str, Any] = {}
    owner_only = {"access", "access_options", "usage"} & set(args.include)
    if owner_only and not ref.owned:
        sections["permission_note"] = "access, access_options and usage are only shown to the owner."
    elif owner_only:
        try:
            if "access" in args.include:
                sections["access"] = await _access_section(ctx, ref.group_id)
            if "access_options" in args.include:
                options = await ctx.api.request("GET", f"{ADMIN}/groups/{ref.group_id}/access/options")
                sections["access_options"] = {
                    "users": [f"user:{u['username']}" for u in options.get("users") or []],
                    "groups": [f"group:{g.get('name') or g['id']}" for g in options.get("groups") or []],
                }
            if "usage" in args.include:
                sections["usage"] = await _usage_section(ctx, ref.group_id, args.usage_window_hours)
        except LensApiError as error:
            if error.status != 403:
                raise
            sections["permission_note"] = "Your Lens user lacks permission to read access or usage for this service."
    return {**view, **sections}


# --- test ------------------------------------------------------------------------------------------------


class TestServiceArgs(ServiceIdArgs):
    __test__ = False

    requests: int = Field(default=1, ge=1, le=5, description="Number of inference requests to send (1-5).")
    prompt: str = Field(default="Reply with one short word.", max_length=200, description="User message to send.")
    max_tokens: int = Field(default=16, ge=1, le=64, description="Completion token limit per request.")


@tool(
    "test_model_service",
    "[W] Check that your own service is callable: control-plane checks (enabled, a routable member, gateway "
    "ready, visible to you) plus up to 5 small chat requests through the gateway. Needs a service token in the "
    "server environment for the requests; without it only the checks run. Use get_model_service to read state "
    "without sending requests.\n"
    "Side effects: sends inference requests and records usage; resources: a few tokens on the backend; "
    "duration: seconds; reversible: n/a.",
    TestServiceArgs,
    read_only=False,
    profile="serving",
)
async def test_model_service(ctx: ToolContext, args: TestServiceArgs) -> dict[str, Any]:
    ref = await resolve_service(ctx, args.service_id)
    require_owned(ref, args.service_id)
    group, members = await load_service(ctx, ref.group_id)
    base_url = await gateway_base_url(ctx, group.get("clusterId"))
    visible = (await ctx.api.request("GET", "/model-service/models")).get("items") or []
    checks = [
        {"check": "service_enabled", "ok": group.get("status") == "active"},
        {"check": "routable_member", "ok": any(is_routable(m) for m in members)},
        {"check": "gateway_ready", "ok": base_url is not None},
        {"check": "visible_to_you", "ok": any(item.get("id") == group["id"] for item in visible)},
    ]
    result: dict[str, Any] = {"service_id": args.service_id, "checks": checks}
    if base_url is None or not all(check["ok"] for check in checks):
        result["inference"] = {"status": "skipped", "reason": "control-plane checks failed"}
        result["ok"] = False
        result["next"] = "Fix the failed checks (get_model_service shows member health), then test again."
        return result
    token = gateway_token()
    if token is None:
        result["inference"] = {
            "status": "skipped",
            "reason": f"{TOKEN_ENV} is not set for this server. {TOKENS_IN_UI}",
        }
        result["ok"] = True
        result["next"] = "Control-plane checks passed; ask the user to create a service token to test real requests."
        return result
    sent = [await send_chat(base_url, token, group["name"], args.prompt, args.max_tokens) for _ in range(args.requests)]
    succeeded = sum(1 for item in sent if item["ok"])
    result["inference"] = {"status": "done", "sent": len(sent), "succeeded": succeeded, "results": sent}
    result["ok"] = succeeded == len(sent)
    result["next"] = "The service answers through the gateway." if result["ok"] else "Inspect results and get_logs."
    return result


# --- update ----------------------------------------------------------------------------------------------


class MemberAdd(Args):
    deployment_id: str = Field(description="Your own ready llm-d deployment (dep-...) of the same model.")
    weight: int = Field(default=0, ge=0, le=100, description="Initial relative weight; 0 adds it on standby.")


class UpdateServiceArgs(ServiceIdArgs):
    add_members: list[MemberAdd] = Field(default_factory=list, max_length=8, description="Deployments to add.")
    remove_members: list[str] = Field(
        default_factory=list, max_length=8, description="Member deployment ids to detach; the deployments are kept."
    )
    weights: dict[str, int] | None = Field(
        default=None,
        description="Patch of relative weights (0-100) keyed by deployment id; members not listed keep theirs.",
    )
    selection_policy: Literal["random", "round_robin", "weighted", "affinity"] | None = Field(
        default=None, description="Selection policy stored on the service."
    )
    description: str | None = Field(default=None, max_length=500, description="New description.")
    expected_version: str | None = Field(
        default=None, description="version from get_model_service; the update is refused if the service changed since."
    )

    @model_validator(mode="after")
    def _something_to_change(self):
        patch = (self.add_members, self.remove_members, self.weights, self.selection_policy, self.description)
        if not any(value is not None and value != [] for value in patch):
            raise ValueError("give at least one of add_members, remove_members, weights, selection_policy, description")
        if self.weights and any(not 0 <= w <= 100 for w in self.weights.values()):
            raise ValueError("weights must be between 0 and 100")
        return self


async def _plan_update(
    ctx: ToolContext, args: UpdateServiceArgs, group: dict[str, Any], members: list[dict[str, Any]]
) -> dict[str, Any]:
    by_execution = {m["executionId"]: m for m in members}
    labels = owned_deployment_ids(ctx, members)  # execution id -> dep id
    current = {e: int(m.get("weight") or 0) for e, m in by_execution.items()}
    desired = dict(current)

    def member_execution(deployment_id: str, field: str) -> str:
        by_label = {label: execution for execution, label in labels.items()}
        execution = by_label.get(deployment_id) or (deployment_id if deployment_id in by_execution else None)
        if execution is None:
            raise ToolError("invalid_argument", f"{deployment_id} is not a member of this service.", field=field)
        return execution

    added: list[str] = []
    if args.add_members:
        publishable = await _publishable(ctx)
        executions = await _owned_executions(ctx, [item.deployment_id for item in args.add_members], "add_members")
        for item in args.add_members:
            execution = executions[item.deployment_id]
            if execution in by_execution:
                raise ToolError("invalid_argument", f"{item.deployment_id} is already a member.", field="add_members")
            if execution not in publishable:
                raise ToolError(
                    "conflict_state",
                    f"Deployment '{item.deployment_id}' is not ready or is not an llm-d deployment.",
                    field="add_members",
                    hint="Wait for get_deployment ready=true. Plain vLLM deployments cannot be members.",
                )
            added.append(execution)
            labels[execution] = item.deployment_id
            desired[execution] = item.weight
    for deployment_id, weight in (args.weights or {}).items():
        desired[member_execution(deployment_id, "weights")] = weight
    removed = [member_execution(dep, "remove_members") for dep in args.remove_members]
    for execution in removed:
        desired[execution] = 0
    remaining = {e: w for e, w in desired.items() if e not in removed}
    if not remaining:
        raise ToolError(
            "in_use",
            "This would remove every member, leaving callers with nothing to serve them.",
            field="remove_members",
            hint="Use disable_model_service to stop the service, or delete_resources to delete it.",
        )
    if group.get("status") == "active" and sum(remaining.values()) <= 0:
        raise ToolError(
            "invalid_argument",
            "All remaining weights would be 0, so nothing would receive traffic.",
            field="weights",
            hint="Give at least one member a weight above 0, or use disable_model_service.",
        )
    return {"current": current, "desired": desired, "added": added, "removed": removed, "labels": labels}


@tool(
    "update_model_service",
    "[D] Change a service you published: add or detach members, set weights, selection policy or description. The "
    "patch is merged server-side and the full member table is returned. Live traffic changes immediately, "
    "so pass expected_version from get_model_service to avoid overwriting someone else's edit. Use "
    "enable_model_service / disable_model_service to start or stop it.\n"
    "Side effects: changes gateway routing weights and members; resources: none; duration: seconds; "
    "reversible: yes, by the reverse update.",
    UpdateServiceArgs,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile="serving",
)
async def update_model_service(ctx: ToolContext, args: UpdateServiceArgs) -> dict[str, Any]:
    ref = await resolve_service(ctx, args.service_id)
    require_owned(ref, args.service_id)
    group, members = await load_service(ctx, ref.group_id)
    current_version = service_version(group, members)
    if args.expected_version and args.expected_version != current_version:
        raise ToolError(
            "version_conflict",
            f"The service was modified (expected {args.expected_version}, now {current_version}).",
            hint="Call get_model_service again and resubmit the update.",
            current_version=current_version,
        )
    plan = await _plan_update(ctx, args, group, members)
    warnings: list[str] = []
    if plan["desired"] != plan["current"]:
        await _put_traffic(ctx, ref.group_id, plan["current"], plan["desired"], warnings)
    member_ids = {m["executionId"]: m["id"] for m in members}
    for execution_id in plan["removed"]:
        try:
            await ctx.api.request("DELETE", f"{ADMIN}/members/{member_ids[execution_id]}", timeout=120)
        except LensApiError as error:
            if error.status != 404:
                raise
    patch = {
        key: value
        for key, value in (("selectionPolicy", args.selection_policy), ("description", args.description))
        if value is not None
    }
    if patch:
        await lens_write(ctx, "PATCH", f"{ADMIN}/groups/{ref.group_id}", patch, warnings)
    group, members = await load_service(ctx, ref.group_id)
    view = await view_of(ctx, ref, group, members)
    labels = plan["labels"]
    return {
        **view,
        "changes": {
            "added": [labels.get(e, e) for e in plan["added"]],
            "removed": [labels.get(e, e) for e in plan["removed"]],
            "weights": {
                labels.get(e, e): {"from": plan["current"].get(e), "to": weight}
                for e, weight in plan["desired"].items()
                if plan["current"].get(e) != weight and e not in plan["removed"]
            },
            "selection_policy": args.selection_policy,
            "description_changed": args.description is not None,
        },
        "warnings": warnings,
        "next": "Call test_model_service to confirm the new routing.",
    }


# --- enable / disable ------------------------------------------------------------------------------------


async def _set_status(ctx: ToolContext, args: ServiceIdArgs, status: str) -> dict[str, Any]:
    ref = await resolve_service(ctx, args.service_id)
    require_owned(ref, args.service_id)
    warnings: list[str] = []
    await lens_write(ctx, "POST", f"{ADMIN}/groups/{ref.group_id}/status", {"status": status}, warnings)
    group, members = await load_service(ctx, ref.group_id)
    view = await view_of(ctx, ref, group, members)
    note = "Callers now receive errors from this endpoint." if status == "disabled" else "Routing resumes shortly."
    return {**view, "warnings": warnings, "next": f"{note} Verify with test_model_service."}


@tool(
    "enable_model_service",
    "[W] Start serving a service you published that was disabled; members, weights and pods are unchanged. "
    "Use disable_model_service to stop it.\n"
    "Side effects: re-enables the gateway route; resources: none; duration: seconds; reversible: yes.",
    ServiceIdArgs,
    read_only=False,
    idempotent=True,
    profile="serving",
)
async def enable_model_service(ctx: ToolContext, args: ServiceIdArgs) -> dict[str, Any]:
    return await _set_status(ctx, args, "active")


@tool(
    "disable_model_service",
    "[D] Stop serving a service you published: callers get errors until enable_model_service. Nothing is deleted; "
    "deployments keep running. To remove it for good use delete_resources.\n"
    "Side effects: removes the route from the gateway; resources: none; duration: seconds; reversible: yes.",
    ServiceIdArgs,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile="serving",
)
async def disable_model_service(ctx: ToolContext, args: ServiceIdArgs) -> dict[str, Any]:
    return await _set_status(ctx, args, "disabled")


# --- access ----------------------------------------------------------------------------------------------


def _split_subject(subject: str) -> tuple[str, str]:
    kind, _, name = subject.partition(":")
    return kind, name


def _check_subjects(subjects: list[str]) -> None:
    for subject in subjects:
        if not SUBJECT_PATTERN.fullmatch(subject):
            raise ValueError(f"subject '{subject}' must look like user:<name> or group:<name>")


class GrantArgs(ServiceIdArgs):
    subjects: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Recipients as 'user:<username or id>' or 'group:<name or id>'. Candidates come from "
        "get_model_service(include=['access_options']); do not guess.",
    )

    @model_validator(mode="after")
    def _valid_subjects(self):
        _check_subjects(self.subjects)
        return self


def _names(item: dict[str, Any]) -> set[str]:
    return {str(item.get(key) or "").lower() for key in ("id", "username", "name", "subjectId", "subjectName")} - {""}


@tool(
    "grant_model_service_access",
    "[W] Allow users or groups to call (inference only) a service you published. Recipients must be eligible; see "
    "get_model_service(include=['access_options']). Use revoke_model_service_access to take access back. "
    "Grants never include management rights.\n"
    "Side effects: adds access grants; resources: none; duration: synchronous; "
    "reversible: revoke_model_service_access.",
    GrantArgs,
    read_only=False,
    idempotent=True,
    open_world=False,
    profile="serving",
)
async def grant_model_service_access(ctx: ToolContext, args: GrantArgs) -> dict[str, Any]:
    ref = await resolve_service(ctx, args.service_id)
    require_owned(ref, args.service_id)
    options = await ctx.api.request("GET", f"{ADMIN}/groups/{ref.group_id}/access/options")
    results = []
    for subject in args.subjects:
        kind, name = _split_subject(subject)
        candidates = options.get("users" if kind == "user" else "groups") or []
        match = next((c for c in candidates if name.lower() in _names(c)), None)
        if match is None:
            results.append({"subject": subject, "status": "not_eligible"})
            continue
        try:
            created = await ctx.api.request(
                "POST", f"{ADMIN}/groups/{ref.group_id}/access", body={"subjectType": kind, "subjectId": match["id"]}
            )
            results.append({"subject": subject, "status": "granted", "binding_id": (created or {}).get("id")})
        except LensApiError as error:
            if error.status != 409:
                raise
            results.append({"subject": subject, "status": "already_granted"})
    if all(item["status"] == "not_eligible" for item in results):
        raise ToolError(
            "invalid_argument",
            "None of the subjects can be granted access to this service.",
            field="subjects",
            hint="Use get_model_service(include=['access_options']) for eligible users and groups.",
        )
    return {
        "service_id": args.service_id,
        "results": results,
        "next": "Recipients still need a service token from the Lens UI.",
    }


class RevokeArgs(Args):
    service_id: str | None = Field(default=None, description="Service id; required when revoking subjects.")
    subjects: list[str] = Field(
        default_factory=list, max_length=20, description="'user:<name or id>' or 'group:<name or id>' grants to remove."
    )
    token_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Your own token ids from get_model_service(include=['access']). Tokens are account-wide, "
        "not specific to one service.",
    )

    @model_validator(mode="after")
    def _valid(self):
        if not self.subjects and not self.token_ids:
            raise ValueError("give subjects or token_ids")
        if self.subjects and not self.service_id:
            raise ValueError("service_id is required when revoking subjects")
        _check_subjects(self.subjects)
        return self


@tool(
    "revoke_model_service_access",
    "[D] Take back access: remove user/group grants from a service you published (by subjects) and/or revoke "
    "your own access tokens (by token_ids). New requests from the revoked parties fail; running streams finish. "
    "To stop the whole service use disable_model_service.\n"
    "Side effects: deletes grants and revokes tokens; resources: none; duration: synchronous; "
    "reversible: grants yes (grant_model_service_access), tokens no.",
    RevokeArgs,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile="serving",
)
async def revoke_model_service_access(ctx: ToolContext, args: RevokeArgs) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if args.subjects:
        service_id = args.service_id or ""
        ref = await resolve_service(ctx, service_id)
        require_owned(ref, service_id)
        grants = (await ctx.api.request("GET", f"{ADMIN}/groups/{ref.group_id}/access")).get("items") or []
        for subject in args.subjects:
            kind, name = _split_subject(subject)
            match = next((g for g in grants if g["subjectType"] == kind and name.lower() in _names(g)), None)
            if match is None:
                results.append({"subject": subject, "status": "not_granted"})
                continue
            await ctx.api.request("DELETE", f"{ADMIN}/groups/{ref.group_id}/access/{match['id']}")
            results.append({"subject": subject, "status": "revoked"})
    for token_id in args.token_ids:
        try:
            await ctx.api.request("DELETE", f"/model-service/tokens/{token_id}")
            results.append({"token_id": token_id, "status": "revoked"})
        except LensApiError as error:
            if error.status != 404:
                raise
            results.append({"token_id": token_id, "status": "not_found"})
    return {"results": results, "next": "Check get_model_service(include=['access']) for the remaining grants."}
