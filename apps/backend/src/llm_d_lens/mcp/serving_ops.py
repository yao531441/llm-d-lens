"""Lens model-service REST calls, ownership checks and the model_service resource kind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .kinds import owner_state, register_kind
from .lens_api import LensApiError
from .path_ids import check_id
from .serving_view import connection_examples, is_routable, service_status, service_view

ADMIN = "/model-service/admin"


@dataclass(frozen=True)
class ServiceRef:
    group_id: str
    ownership: str  # owned / shared / unknown, from the group's ownerUserId

    @property
    def owned(self) -> bool:
        return self.ownership == "owned"


def service_not_found(service_id: str) -> ToolError:
    return ToolError(
        "not_found",
        f"No model service '{service_id}' is visible to you.",
        field="service_id",
        hint="Call resource_list(kinds=['model_service'], owner='all') to see valid ids.",
    )


async def get_group(ctx: ToolContext, group_id: str) -> dict[str, Any] | None:
    try:
        return await ctx.api.request("GET", f"{ADMIN}/groups/{group_id}")
    except LensApiError as error:
        if error.status == 404:
            return None
        raise


async def resolve_service(ctx: ToolContext, service_id: str) -> ServiceRef:
    check_id(service_id, "service_id")
    group = await get_group(ctx, service_id)
    if group is None:
        raise service_not_found(service_id)
    return ServiceRef(service_id, owner_state(ctx, group.get("ownerUserId")))


def require_owned(ref: ServiceRef, service_id: str) -> None:
    """The agent-only ownership rule: Lens RBAC alone would let any publisher change every service."""
    if ref.owned:
        return
    raise ToolError(
        "not_owner" if ref.ownership == "shared" else "owner_unknown",
        f"{service_id} "
        + ("belongs to another user." if ref.ownership == "shared" else "has no recorded owner.")
        + " Agents cannot change services they do not own.",
        field="service_id",
        hint="Use resource_list(kinds=['model_service']) to see the services you own.",
    )


async def load_service(ctx: ToolContext, group_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group = await get_group(ctx, group_id)
    if group is None:
        raise service_not_found(group_id)
    members = (await ctx.api.request("GET", f"{ADMIN}/members", params={"group_id": group_id})).get("items") or []
    return group, members


async def gateway_base_url(ctx: ToolContext, cluster_id: str | None) -> str | None:
    connection = await ctx.api.request("GET", "/model-service/connection")
    for item in connection.get("clusters") or []:
        if item.get("clusterId") == cluster_id and item.get("ready") and item.get("baseUrl"):
            return item["baseUrl"].rstrip("/")
    return None


async def view_of(
    ctx: ToolContext, ref: ServiceRef, group: dict[str, Any], members: list[dict[str, Any]]
) -> dict[str, Any]:
    return service_view(
        group, members, base_url=await gateway_base_url(ctx, group.get("clusterId")), ownership=ref.ownership
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
                "The change was saved but the gateway routing could not be updated yet. Check model_service_get later; "
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
            hint="Call model_service_get again and resubmit the update.",
        )
    if "plain vllm" in lowered or "model name must match" in lowered or "distinct" in lowered:
        return ToolError(
            "unsupported_combination", message, hint="Members must be llm-d deployments of the same model."
        )
    if "already" in lowered or "duplicate" in lowered or "unique" in lowered:
        return ToolError("already_exists", message, hint="Choose another public_name or use model_service_update.")
    return ToolError("conflict_state", message, hint="Check the deployments with deployment_get and retry when ready.")


# --- model_service kind: used by resource_list and resource_delete ---------------------------------


def _entry(ctx: ToolContext, group: dict[str, Any], members: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    entry = {
        "id": group["id"],
        "kind": "model_service",
        "name": group["name"],
        "status": service_status(group, members) if members is not None else group.get("status"),
        "ownership": owner_state(ctx, group.get("ownerUserId")),
        "cluster_id": group.get("clusterId"),
        "created_at": group.get("createdAt"),
    }
    if members is not None:
        entry["members"] = len(members)
        entry["routable_members"] = sum(1 for m in members if is_routable(m))
    return entry


async def list_services(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    groups = (await ctx.api.request("GET", f"{ADMIN}/groups")).get("items") or []
    return [_entry(ctx, g) for g in groups if not cluster_id or g.get("clusterId") == cluster_id]


async def get_service(ctx: ToolContext, service_id: str) -> dict[str, Any] | None:
    check_id(service_id, "id")
    group = await get_group(ctx, service_id)
    if group is None:
        return None
    members = (await ctx.api.request("GET", f"{ADMIN}/members", params={"group_id": service_id})).get("items") or []
    return _entry(ctx, group, members)


async def delete_service(ctx: ToolContext, service_id: str) -> list[dict[str, str]]:
    """Detach the member deployments (they are kept) and delete the group, which also drops its access grants."""
    members = (await ctx.api.request("GET", f"{ADMIN}/members", params={"group_id": service_id})).get("items") or []
    detached = []
    for member in members:
        try:
            await ctx.api.request("DELETE", f"{ADMIN}/members/{member['id']}", timeout=120)
        except LensApiError as error:
            if error.status != 404:
                raise
        detached.append({"kind": "model_service_member", "id": member["executionId"]})
    try:
        await ctx.api.request("DELETE", f"{ADMIN}/groups/{service_id}", timeout=120)
    except LensApiError as error:
        if error.status != 404:
            raise
    return detached


register_kind("model_service", list=list_services, get=get_service, delete=delete_service)
