"""Cluster-admin profile: shared components, image prepull, node scheduling, the model gateway and orphan cleanup."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..admin_components import install_component, operation_progress, plan_component, prepull_state
from ..admin_gateway import configure_gateway, reconcile_gateway
from ..admin_guard import CLUSTER_ID_PATTERN
from ..admin_orphans import delete_orphans, list_orphans
from ..context import ToolContext
from ..errors import ToolError
from ..registry import tool

PROFILE = "cluster-admin"
NODE_NAME_PATTERN = r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$"

ClusterId = Annotated[str, Field(pattern=CLUSTER_ID_PATTERN, description="Cluster id from list_clusters.")]
NodeName = Annotated[str, Field(pattern=NODE_NAME_PATTERN)]
Component = Literal[
    "gateway-api-crds", "lws", "monitoring-stack", "accelerator-monitoring", "model-gateway", "software-downloads"
]


class PlanClusterComponentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    component: Component | Literal["image-prepull", "orphan-namespaces"] = Field(
        description="Component to inspect. 'image-prepull' reports the prepull progress and 'orphan-namespaces' lists "
        "the namespaces Lens considers orphaned."
    )
    accelerator: Literal["intel_gpu", "nvidia_gpu"] = Field(
        "intel_gpu", description="accelerator-monitoring only: which exporter to inspect."
    )
    access_mode: Literal["dra", "plugin"] = Field("dra", description="accelerator-monitoring only.")
    operation_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Progress of an install started by install_cluster_component (monitoring-stack or "
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
    accelerators: list[Literal["intel", "nvidia"]] = Field(
        min_length=1, max_length=2, description="Accelerator families whose model-server images are pulled."
    )


class NodesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: ClusterId
    nodes: list[NodeName] = Field(
        min_length=1,
        max_length=64,
        description="Node names from get_cluster. Control-plane nodes are refused by Lens.",
    )
    reason: str | None = Field(None, max_length=300, description="Recorded in the audit log.")


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
        description="Orphan namespaces to delete, taken from plan_cluster_component(component='orphan-namespaces').",
    )


@tool(
    "plan_cluster_component",
    "[R] Check a shared cluster component before installing or upgrading it: whether it is installed, Lens's "
    "preflight checks and warnings. Also reports image-prepull progress, lists orphan namespaces, and with "
    "operation_id follows a running install. Does not change the cluster; apply with install_cluster_component.\n"
    "Side effects: none; resources: none; duration: synchronous, seconds; reversible: n/a.",
    PlanClusterComponentInput,
    read_only=True,
    open_world=True,
    profile=PROFILE,
)
async def plan_cluster_component(ctx: ToolContext, args: PlanClusterComponentInput) -> dict[str, Any]:
    result: dict[str, Any] = {"cluster_id": args.cluster_id, "component": args.component}
    if args.operation_id:
        return {**result, **await operation_progress(ctx, args.component, args.accelerator, args.operation_id)}
    if args.component == "orphan-namespaces":
        return {**result, **await list_orphans(ctx, args.cluster_id)}
    result.update(await plan_component(ctx, args.cluster_id, args.component, args.accelerator, args.access_mode))
    if result.get("installed") is False or result.get("allowed") is True:
        result["next"] = "Apply with install_cluster_component if this is intended."
    return result


@tool(
    "install_cluster_component",
    "[D] Install or upgrade a shared cluster component (Gateway API CRDs, LeaderWorkerSet, monitoring stack, "
    "accelerator monitoring, model gateway, llm-d software sources) to the version Lens pins. Upgrades can disturb "
    "everyone using the cluster. Run plan_cluster_component first. Accelerator drivers are never installed here.\n"
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
        result["next"] = "Follow it with plan_cluster_component(operation_id=...)."
    return result


@tool(
    "prepull_images",
    "[W] Start pulling the llm-d router and model-server images for the chosen accelerator families onto every node, "
    "so later deployments start faster. Progress is shown by plan_cluster_component(component='image-prepull').\n"
    "Side effects: runs a puller DaemonSet in Lens's images namespace; resources: node disk and registry bandwidth; "
    "duration: asynchronous, minutes; reversible: yes.",
    PrepullImagesInput,
    read_only=False,
    idempotent=True,
    open_world=True,
    profile=PROFILE,
)
async def prepull_images(ctx: ToolContext, args: PrepullImagesInput) -> dict[str, Any]:
    status = await ctx.api.request(
        "POST",
        f"/cluster/clusters/{args.cluster_id}/images/prepull",
        body={"accelerators": list(dict.fromkeys(args.accelerators))},
        timeout=120,
    )
    return {
        "cluster_id": args.cluster_id,
        **prepull_state(status),
        "next": "Check progress with plan_cluster_component(component='image-prepull').",
    }


async def _schedule(ctx: ToolContext, args: NodesInput, action: Literal["cordon", "uncordon"]) -> dict[str, Any]:
    outcome = await ctx.api.request(
        "POST", f"/cluster/clusters/{args.cluster_id}/nodes/{action}", body={"names": args.nodes}, timeout=120
    )
    items = outcome.get("items") or []
    failed = [item for item in items if not item.get("ok")]
    result: dict[str, Any] = {
        "cluster_id": args.cluster_id,
        "done": [item["name"] for item in items if item.get("ok")],
        "failed": [item.get("name") for item in failed],
    }
    if failed:
        result["untrusted_text"] = [f"{item.get('name')}: {item.get('error')}" for item in failed]
    return result


@tool(
    "cordon_nodes",
    "[D] Mark nodes unschedulable so no new pods land on them; running pods are NOT evicted (no drain). Use it to "
    "keep work off a node during maintenance, and undo it with uncordon_nodes. Control-plane nodes are refused.\n"
    "Side effects: sets spec.unschedulable on the nodes; resources: reduces schedulable capacity; duration: "
    "synchronous; reversible: yes, uncordon_nodes.",
    NodesInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def cordon_nodes(ctx: ToolContext, args: NodesInput) -> dict[str, Any]:
    return await _schedule(ctx, args, "cordon")


@tool(
    "uncordon_nodes",
    "[W] Make cordoned nodes schedulable again. Nodes that are already schedulable are left unchanged.\n"
    "Side effects: clears spec.unschedulable on the nodes; resources: restores schedulable capacity; duration: "
    "synchronous; reversible: yes, cordon_nodes.",
    NodesInput,
    read_only=False,
    idempotent=True,
    profile=PROFILE,
)
async def uncordon_nodes(ctx: ToolContext, args: NodesInput) -> dict[str, Any]:
    return await _schedule(ctx, args, "uncordon")


@tool(
    "configure_model_gateway",
    "[D] Change the shared model gateway of a cluster: data-plane replicas and/or the inference payload processor "
    "(IPP) config, which is replaced whole and reinstalled. Requests in flight can be disturbed. Only the gateway "
    "Lens installed is touched. To rewrite routes from Lens's records use reconcile_model_gateway.\n"
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
    "reconcile_model_gateway",
    "[D] Rewrite the gateway and model routes of every cluster that has active model-service members so they match "
    "Lens's records. Lens schedules this for all such clusters at once and returns immediately; it cannot be limited "
    "to one. Use configure_model_gateway for replica or IPP changes.\n"
    "Side effects: re-applies Gateway and HTTPRoute objects; resources: shared gateways on all clusters with "
    "members; duration: asynchronous; reversible: no.",
    ReconcileModelGatewayInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def reconcile_model_gateway(ctx: ToolContext, args: ReconcileModelGatewayInput) -> dict[str, Any]:
    return {**await reconcile_gateway(ctx), "next": "Check progress with get_cluster or the gateway status in Lens."}


@tool(
    "delete_orphan_resources",
    "[D] Delete namespaces that Lens lists as orphaned (no Lens deployment tracks them), by explicit name together "
    "with the created_at Lens shows for it. Anything not currently listed as an orphan, or whose created_at differs, "
    "refuses the whole call. Protected namespaces are never touched. List candidates with "
    "plan_cluster_component(component='orphan-namespaces'); for your own deployments use delete_resources.\n"
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
