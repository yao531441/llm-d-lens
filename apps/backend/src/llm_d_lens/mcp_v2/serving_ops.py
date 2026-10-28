"""Lens model-service REST calls, ownership checks and the model_service resource kind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .kinds import register_kind
from .ledger import Resource
from .lens_api import LensApiError
from .serving_view import connection_examples, is_routable, service_status, service_view

ADMIN = "/model-service/admin"


@dataclass(frozen=True)
class ServiceRef:
    row: Resource | None  # ledger record when the caller created the service
    group_id: str
    foreign_owner: bool = False

    @property
    def owned(self) -> bool:
        return self.row is not None and self.row.state in ("active", "deleting")


def is_definite_failure(error: BaseException) -> bool:
    """True when Lens clearly refused the request, so nothing was created."""
    if isinstance(error, ToolError):
        return True
    return isinstance(error, LensApiError) and 400 <= error.status < 500 and error.status not in (408, 429)


def outcome_unknown(row: Resource, error: BaseException) -> ToolError:
    cause = f"HTTP {error.status}" if isinstance(error, LensApiError) else type(error).__name__
    return ToolError(
        "upstream_unavailable",
        f"Lens did not confirm whether service '{row.name}' was published ({cause}); it may exist.",
        hint=f"Wait a moment, then call get_model_service(service_id='{row.id}'); it reconciles by name. "
        "Do not publish the same name again before that.",
        service_id=row.id,
    )


async def reconcile_service(ctx: ToolContext, row: Resource) -> Resource:
    """Bind an unconfirmed ledger row to the Lens group with its name, if Lens has one."""
    if row.lens_execution_id or row.state not in ("creating", "deleting"):
        return row
    try:
        groups = (await ctx.api.request("GET", f"{ADMIN}/groups")).get("items") or []
    except Exception:  # noqa: BLE001 - still unknown; the caller reports it
        return row
    for group in groups:
        same = (
            group.get("clusterId") == row.cluster_id and str(group.get("name", "")).lower() == (row.name or "").lower()
        )
        if same and group.get("ownerUserId") in (None, ctx.owner):
            ctx.ledger.bind_execution(row.id, group["id"])
            if row.state == "creating":
                ctx.ledger.set_state(row.id, "active")
            return ctx.ledger.get(row.id) or row
    return row


def service_not_found(service_id: str) -> ToolError:
    return ToolError(
        "not_found",
        f"No model service '{service_id}' is visible to you.",
        field="service_id",
        hint="Call list_resources(kinds=['model_service'], owner='all') to see valid ids.",
    )


async def resolve_service(ctx: ToolContext, service_id: str) -> ServiceRef:
    """Accepts a ledger id (``svc-...``) or a raw Lens group id; ownership comes only from the ledger."""
    if service_id.startswith("svc-"):
        row = ctx.ledger.get(service_id)
        if row is None or row.kind != "model_service" or row.state == "deleted":
            raise service_not_found(service_id)
        if not row.lens_execution_id and row.owner == ctx.owner:
            row = await reconcile_service(ctx, row)
        if not row.lens_execution_id:
            raise ToolError(
                "conflict_state",
                f"{service_id} was being published but Lens has not confirmed it, so it may not exist.",
                field="service_id",
                hint="Retry shortly. If it stays unconfirmed, remove the record with delete_resources.",
            )
        if row.owner != ctx.owner:
            return ServiceRef(None, row.lens_execution_id, foreign_owner=True)
        return ServiceRef(row, row.lens_execution_id)
    row = ctx.ledger.find_by_execution(service_id)
    if row is not None and (row.kind != "model_service" or row.owner != ctx.owner):
        row = None
    return ServiceRef(row, service_id)


def require_owned(ref: ServiceRef, service_id: str) -> Resource:
    if ref.row is not None and ref.owned:
        return ref.row
    if ref.foreign_owner:
        raise ToolError(
            "not_owner",
            f"{service_id} belongs to another user. Agents cannot change other people's services.",
            field="service_id",
            hint="Use list_resources(kinds=['model_service']) to see the services you own.",
        )
    raise ToolError(
        "owner_unknown",
        f"{service_id} was not published through this server, so its owner cannot be verified.",
        field="service_id",
        hint="Only services you published with publish_model_service can be changed here. Ask the owner or an admin.",
    )


async def load_service(ctx: ToolContext, group_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        group = await ctx.api.request("GET", f"{ADMIN}/groups/{group_id}")
    except LensApiError as error:
        if error.status == 404:
            raise service_not_found(group_id) from error
        raise
    members = (await ctx.api.request("GET", f"{ADMIN}/members", params={"group_id": group_id})).get("items") or []
    return group, members


async def gateway_base_url(ctx: ToolContext, cluster_id: str | None) -> str | None:
    connection = await ctx.api.request("GET", "/model-service/connection")
    for item in connection.get("clusters") or []:
        if item.get("clusterId") == cluster_id and item.get("ready") and item.get("baseUrl"):
            return item["baseUrl"].rstrip("/")
    return None


def owned_deployment_ids(ctx: ToolContext, members: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {}
    for member in members:
        row = ctx.ledger.find_by_execution(member["executionId"])
        if row is not None and row.kind == "deployment" and row.owner == ctx.owner:
            mapping[member["executionId"]] = row.id
    return mapping


async def view_of(
    ctx: ToolContext, ref: ServiceRef, group: dict[str, Any], members: list[dict[str, Any]]
) -> dict[str, Any]:
    return service_view(
        group,
        members,
        service_id=ref.row.id if ref.row else None,
        deployment_ids=owned_deployment_ids(ctx, members),
        base_url=await gateway_base_url(ctx, group.get("clusterId")),
        owned=ref.owned,
    )


def connection_info(view: dict[str, Any]) -> dict[str, Any] | None:
    endpoint = view.get("endpoint")
    return connection_examples(endpoint["base_url"], endpoint["model"]) if endpoint else None


APPLY_FAILED = {"model_publication_apply_failed", "model_traffic_apply_failed"}


async def lens_write(ctx: ToolContext, method: str, path: str, body: Any, warnings: list[str]) -> Any:
    """Write to Lens. A 502 'saved but routing not applied' becomes a warning because the change is persisted."""
    try:
        return await ctx.api.request(method, path, body=body, timeout=120)
    except LensApiError as error:
        if error.code() in APPLY_FAILED:
            warnings.append(
                "The change was saved but the gateway routing could not be updated yet. Check get_model_service later; "
                "if it stays degraded an administrator can reconcile the gateway."
            )
            return None
        raise conflict_error(error) from error


def conflict_error(error: LensApiError) -> LensApiError | ToolError:
    """Explain Lens 409 conflicts with the codes agents act on; everything else propagates unchanged."""
    if error.status != 409:
        return error
    message = error.message()
    lowered = message.lower()
    if error.code() == "model_traffic_conflict" and (
        "expected" in lowered or "stale" in lowered or "changed" in lowered
    ):
        return ToolError(
            "version_conflict",
            "The service was changed by someone else while this update was being applied.",
            hint="Call get_model_service again and resubmit the update.",
        )
    if "plain vllm" in lowered or "model name must match" in lowered or "distinct" in lowered:
        return ToolError(
            "unsupported_combination", message, hint="Members must be llm-d deployments of the same model."
        )
    if "already" in lowered or "duplicate" in lowered or "unique" in lowered:
        return ToolError("already_exists", message, hint="Choose another public_name or use update_model_service.")
    return ToolError("conflict_state", message, hint="Check the deployments with get_deployment and retry when ready.")


# --- model_service kind: used by list_resources and delete_resources ---------------------------------


async def describe(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    row = await reconcile_service(ctx, row)
    if not row.lens_execution_id:
        return {"live_status": "unconfirmed"}
    try:
        group, members = await load_service(ctx, row.lens_execution_id or "")
    except ToolError:
        return {"live_status": "missing"}
    return {
        "live_status": service_status(group, members),
        "public_name": group["name"],
        "members": len(members),
        "routable_members": sum(1 for m in members if is_routable(m)),
    }


async def dependents(ctx: ToolContext, row: Resource) -> list[dict[str, Any]]:
    row = await reconcile_service(ctx, row)
    if not row.lens_execution_id:
        return []
    try:
        group, members = await load_service(ctx, row.lens_execution_id or "")
    except ToolError:
        return []
    mapping = owned_deployment_ids(ctx, members)
    items = [
        {
            "id": mapping.get(m["executionId"], m["executionId"]),
            "kind": "deployment",
            "name": m.get("name"),
            "reason": "serves traffic for this service; it is detached on delete and the deployment itself is kept",
        }
        for m in members
    ]
    try:
        grants = await ctx.api.request("GET", f"{ADMIN}/groups/{group['id']}/access")
    except LensApiError:
        grants = {}
    if grants.get("items"):
        items.append(
            {
                "id": group["id"],
                "kind": "access_grant",
                "name": f"{len(grants['items'])} grant(s)",
                "reason": "callers lose access to this endpoint",
            }
        )
    return items


async def delete(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    row = await reconcile_service(ctx, row)
    group_id = row.lens_execution_id
    if not group_id:
        return
    try:
        members = (await ctx.api.request("GET", f"{ADMIN}/members", params={"group_id": group_id})).get("items") or []
    except LensApiError as error:
        if error.status == 404:
            return
        raise
    if members and not cascade:
        raise ToolError(
            "in_use",
            f"The service still has {len(members)} member deployment(s) serving callers.",
            hint="Remove them with update_model_service(remove_members) or pass cascade=true to delete_resources.",
        )
    for member in members:
        try:
            await ctx.api.request("DELETE", f"{ADMIN}/members/{member['id']}", timeout=120)
        except LensApiError as error:
            if error.status != 404:
                raise
    try:
        await ctx.api.request("DELETE", f"{ADMIN}/groups/{group_id}", timeout=120)
    except LensApiError as error:
        if error.status != 404:
            raise


async def list_others(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    groups = (await ctx.api.request("GET", f"{ADMIN}/groups")).get("items") or []
    entries = []
    for group in groups:
        if cluster_id and group.get("clusterId") != cluster_id:
            continue
        known = ctx.ledger.find_by_execution(group["id"])
        entries.append(
            {
                "id": group["id"],
                "lens_id": group["id"],
                "kind": "model_service",
                "name": group["name"],
                "ownership": "shared" if known is not None else "unknown",
                "state": group.get("status"),
                "cluster_id": group.get("clusterId"),
            }
        )
    return entries


register_kind("model_service", delete=delete, describe=describe, dependents=dependents, list_others=list_others)
