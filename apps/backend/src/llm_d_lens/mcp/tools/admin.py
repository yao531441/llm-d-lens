"""Cluster-admin profile: shared components, image prepull, the model gateway and orphan namespaces."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..admin_components import check_component, install_component, operation_progress, prepull_state
from ..admin_gateway import configure_gateway, reconcile_gateway
from ..admin_guard import CLUSTER_ID_PATTERN
from ..admin_orphans import delete_orphans, list_orphans
from ..context import ToolContext
from ..errors import ToolError
from ..registry import tool

PROFILE = "cluster-admin"

ClusterId = Annotated[str, Field(pattern=CLUSTER_ID_PATTERN, description="Cluster id from cluster_list.")]
Component = Literal[
    "gateway-api-crds", "lws", "monitoring-stack", "accelerator-monitoring", "model-gateway", "software-downloads"
]


class CheckClusterComponentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    component: Component = Field(description="Component to inspect.")
    accelerator: Literal["intel_gpu", "nvidia_gpu"] = Field(
        "intel_gpu", description="accelerator-monitoring only: which exporter to inspect."
    )
    access_mode: Literal["dra", "plugin"] = Field("dra", description="accelerator-monitoring only.")
    operation_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Progress of an install started by cluster_component_install (monitoring-stack or "
        "accelerator-monitoring) instead of a fresh preflight.",
    )


class InstallClusterComponentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    component: Component = Field(description="Component to install or upgrade to the version pinned by Lens.")
    accelerator: Literal["intel_gpu", "nvidia_gpu"] = Field(
        "intel_gpu", description="accelerator-monitoring only: which exporter to install."
    )
    access_mode: Literal["dra", "plugin"] = Field("dra", description="accelerator-monitoring only.")
    reinstall: bool = Field(False, description="monitoring-stack only: reinstall even if the release is current.")


class PrepullImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    accelerators: list[Literal["intel", "nvidia"]] | None = Field(
        None,
        min_length=1,
        max_length=2,
        description="Accelerator families whose model-server images are pulled. Omit to only read the progress of "
        "the current prepull.",
    )


class ListOrphanResourcesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId


class ConfigureModelGatewayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    replicas: int | None = Field(None, ge=1, le=20, description="Gateway data-plane replicas.")
    ipp_config: str | None = Field(
        None,
        max_length=20000,
        description="Full PayloadProcessorConfig YAML for the inference payload processor; it replaces the live one.",
    )


class ReconcileModelGatewayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrphanNamespace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=63, description="Namespace name exactly as listed.")
    created_at: str = Field(
        description="created_at exactly as listed; protects against a newer namespace reusing the name."
    )


class DeleteOrphanResourcesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    namespaces: list[OrphanNamespace] = Field(
        min_length=1,
        max_length=20,
        description="Orphan namespaces to delete, taken from orphan_resource_list.",
    )


@tool(
    "cluster_component_check",
    "[R] Check a shared cluster component before installing or upgrading it: whether it is installed, Lens's "
    "preflight checks and warnings. With operation_id it follows a running install. Does not change the cluster; "
    "apply with cluster_component_install.\n"
    "Side effects: none; resources: none; duration: synchronous, seconds; reversible: n/a.",
    CheckClusterComponentInput,
    read_only=True,
    open_world=True,
    profile=PROFILE,
)
async def check_cluster_component(ctx: ToolContext, args: CheckClusterComponentInput) -> dict[str, Any]:
    result: dict[str, Any] = {"cluster_id": args.cluster_id, "component": args.component}
    if args.operation_id:
        return {**result, **await operation_progress(ctx, args.component, args.accelerator, args.operation_id)}
    result.update(await check_component(ctx, args.cluster_id, args.component, args.accelerator, args.access_mode))
    if result.get("installed") is False or result.get("allowed") is True:
        result["next"] = "Apply with cluster_component_install if this is intended."
    return result


@tool(
    "cluster_component_install",
    "[D] Install or upgrade a shared cluster component (Gateway API CRDs, LeaderWorkerSet, monitoring stack, "
    "accelerator monitoring, model gateway, llm-d software sources) to the version Lens pins. Upgrades can disturb "
    "everyone using the cluster. Run cluster_component_check first. Accelerator drivers are never installed here.\n"
    "Side effects: applies CRDs and Helm releases in the component's own namespace; resources: cluster-wide "
    "components; duration: synchronous for CRDs, LWS and gateway (minutes), asynchronous for monitoring; "
    "reversible: no.",
    InstallClusterComponentInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    open_world=True,
    profile=PROFILE,
)
async def install_cluster_component(ctx: ToolContext, args: InstallClusterComponentInput) -> dict[str, Any]:
    result = await install_component(
        ctx, args.cluster_id, args.component, args.accelerator, args.access_mode, args.reinstall
    )
    result = {"cluster_id": args.cluster_id, "component": args.component, **result}
    if result.get("operation_id") and args.component in {"monitoring-stack", "accelerator-monitoring"}:
        result["next"] = "Follow it with cluster_component_check(operation_id=...)."
    return result


@tool(
    "image_prepull",
    "[W] Start pulling the llm-d router and model-server images for the chosen accelerator families onto every node, "
    "so later deployments start faster, or, without accelerators, read the progress of the current prepull.\n"
    "Side effects: runs a puller DaemonSet in Lens's images namespace; resources: node disk and registry bandwidth; "
    "duration: asynchronous, minutes; reversible: yes.",
    PrepullImagesInput,
    read_only=False,
    idempotent=True,
    open_world=True,
    profile=PROFILE,
)
async def prepull_images(ctx: ToolContext, args: PrepullImagesInput) -> dict[str, Any]:
    path = f"/cluster/clusters/{args.cluster_id}/images/prepull"
    if args.accelerators is None:
        return {"cluster_id": args.cluster_id, **prepull_state(await ctx.api.request("GET", path))}
    status = await ctx.api.request(
        "POST", path, body={"accelerators": list(dict.fromkeys(args.accelerators))}, timeout=120
    )
    return {
        "cluster_id": args.cluster_id,
        **prepull_state(status),
        "next": "Read the progress with image_prepull(cluster_id=...) without accelerators.",
    }


@tool(
    "model_gateway_configure",
    "[D] Change the shared model gateway of a cluster: data-plane replicas and/or the inference payload processor "
    "(IPP) config, which is replaced whole and reinstalled. Requests in flight can be disturbed. Only the gateway "
    "Lens installed is touched. To rewrite routes from Lens's records use model_gateway_reconcile.\n"
    "Side effects: scales the gateway Deployment and upgrades the IPP release; resources: the shared gateway; "
    "duration: synchronous, up to a few minutes; reversible: yes, by configuring the previous values.",
    ConfigureModelGatewayInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def configure_model_gateway(ctx: ToolContext, args: ConfigureModelGatewayInput) -> dict[str, Any]:
    if args.replicas is None and args.ipp_config is None:
        raise ToolError("invalid_argument", "Give replicas, ipp_config or both.", field="replicas")
    return await configure_gateway(ctx, args.cluster_id, args.replicas, args.ipp_config)


@tool(
    "model_gateway_reconcile",
    "[D] Rewrite the gateway and model routes of every cluster that has active model-service members so they match "
    "Lens's records. Lens schedules this for all such clusters at once and returns immediately; it cannot be limited "
    "to one. Use model_gateway_configure for replica or IPP changes.\n"
    "Side effects: re-applies Gateway and HTTPRoute objects; resources: shared gateways on all clusters with "
    "members; duration: asynchronous; reversible: no.",
    ReconcileModelGatewayInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def reconcile_model_gateway(ctx: ToolContext, args: ReconcileModelGatewayInput) -> dict[str, Any]:
    return {**await reconcile_gateway(ctx), "next": "Check progress with cluster_get or the gateway status in Lens."}


@tool(
    "orphan_resource_list",
    "[R] List the namespaces Lens considers orphaned on a cluster (no Lens deployment tracks them), with the "
    "created_at that orphan_resource_delete needs.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    ListOrphanResourcesInput,
    read_only=True,
    profile=PROFILE,
)
async def list_orphan_resources(ctx: ToolContext, args: ListOrphanResourcesInput) -> dict[str, Any]:
    return {"cluster_id": args.cluster_id, **await list_orphans(ctx, args.cluster_id)}


@tool(
    "orphan_resource_delete",
    "[D] Delete namespaces that Lens lists as orphaned (no Lens deployment tracks them), by explicit name together "
    "with the created_at Lens shows for it. Anything not currently listed as an orphan, or whose created_at differs, "
    "refuses the whole call. Protected namespaces are never touched. List candidates with "
    "orphan_resource_list; for your own deployments use resource_delete.\n"
    "Side effects: deletes the namespaces and everything in them; resources: frees accelerators and disk; "
    "duration: synchronous, up to minutes; reversible: no.",
    DeleteOrphanResourcesInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def delete_orphan_resources(ctx: ToolContext, args: DeleteOrphanResourcesInput) -> dict[str, Any]:
    return await delete_orphans(ctx, args.cluster_id, [(item.name, item.created_at) for item in args.namespaces])
