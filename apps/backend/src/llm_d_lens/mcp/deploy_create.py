"""Planning and creating one deployment run: the same Lens calls the Web UI makes, with nothing stored in between."""

from __future__ import annotations

import hashlib
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from llm_d_lens.deploy.prepared_plans import requires_prepared_plan

from .cluster_info import cluster_summary
from .context import ToolContext
from .deployment_refs import resolve
from .errors import ToolError
from .evaluation_design import canonical
from .guide_fields import LONE_DEPLOYMENT, GuidePlanFields, normalize_fields
from .planning import plan_configuration, slug

CREATED_VIA = "lens-mcp"


@dataclass
class Prepared:
    cluster_id: str
    session_id: str
    name: str
    configuration: dict[str, Any]  # the deployable configuration Lens will run
    resolved: dict[str, Any]
    gpus: int | None
    free_gpus: int | None
    warnings: list[dict[str, str]] = field(default_factory=list)
    source: dict[str, Any] | None = None
    applied_patches: list[dict[str, Any]] = field(default_factory=list)
    digest: str = ""

    @property
    def needs_review(self) -> bool:
        """Lens deploys a DisaggregatedSet only from a reviewed plan."""
        return requires_prepared_plan([SimpleNamespace(content=self.configuration.get("content") or {})])


def _executable(configuration: dict[str, Any], source: dict[str, Any] | None, chosen: str) -> str:
    """The rendered configuration and guide source; the deployment name is masked so generated names agree."""
    body = {key: value for key, value in configuration.items() if key != "checksum"}
    return canonical({"configuration": body, "source": source}).replace(chosen, "<name>")


def _digest(cluster_id: str, resolved: dict[str, Any], executable: str | None = None) -> str:
    """Over the request, what Lens resolved and the rendered configuration Lens would run.

    The deployment name is not part of it: a plan made with a generated name must match the create that passes
    that name back.
    """
    payload = {"cluster_id": cluster_id, "resolved": resolved, "executable": executable}
    return "sha256:" + hashlib.sha256(canonical(payload).encode()).hexdigest()[:32]


async def _from_configuration(
    ctx: ToolContext, configuration: GuidePlanFields, cluster_id: str, name: str | None, volume_id: str | None
) -> Prepared:
    fields = normalize_fields(configuration.model_dump(by_alias=True, exclude_none=True))
    summary = await cluster_summary(ctx, cluster_id)
    session = await ctx.api.request("GET", "/cluster/session", params={"clusterId": cluster_id})
    catalog = (
        await ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": cluster_id})
        if (fields.get("source") or {}).get("mode", "official") == "official"
        else {}
    )
    chosen = name or f"mcp-{slug(fields['model'].rsplit('/', 1)[-1], 30)}-{uuid.uuid4().hex[:4]}"
    planned = await plan_configuration(
        ctx,
        LONE_DEPLOYMENT,
        fields,
        cluster_id=cluster_id,
        cluster=session["cluster"],
        session_id=session["sessionId"],
        summary=summary,
        catalog=catalog,
        deployment_name=chosen,
        model_cache_volume_id=volume_id,
    )
    resolved = {
        **planned.fields,
        "accelerator": planned.accelerator,
        "modelServer": planned.model_server,
        "guideVariant": planned.guide_variant,
        "runtimeImage": planned.image,
    }
    free = summary.get("accelerators_free")
    warnings = [{"code": "guide_warning", "message": text} for text in planned.warnings]
    if free is not None and planned.gpus > free:
        warnings.append({"code": "insufficient_capacity", "message": "The plan exceeds currently free accelerators."})
    return Prepared(
        cluster_id,
        session["sessionId"],
        chosen,
        planned.configuration,
        resolved,
        planned.gpus,
        free,
        warnings,
        planned.source,
        planned.applied_patches,
        _digest(cluster_id, resolved, _executable(planned.configuration, planned.source, chosen)),
    )


async def _from_deployment(ctx: ToolContext, source_id: str, cluster_id: str | None, name: str | None) -> Prepared:
    """A copy of the Lens configuration of an existing deployment, kept exactly as Lens stored it."""
    execution = (await resolve(ctx, source_id)).execution
    configuration = execution.get("configuration")
    if not configuration:
        raise ToolError(
            "configuration_unavailable",
            "Lens did not return this deployment's source configuration.",
            field="from_deployment_id",
            hint="Plan it again from guide fields with configuration=...",
        )
    if cluster_id and cluster_id != execution.get("cluster_id"):
        raise ToolError(
            "invalid_argument",
            "A copy keeps the cluster and storage binding of the source deployment.",
            field="cluster_id",
            hint="Plan a new configuration for another cluster.",
        )
    cluster_id = execution["cluster_id"]
    session = await ctx.api.request("GET", "/cluster/session", params={"clusterId": cluster_id})
    resolved = {"from_deployment_id": source_id, "configuration": configuration}
    configuration = deepcopy(configuration)
    chosen = name or f"mcp-clone-{uuid.uuid4().hex[:8]}"
    configuration.setdefault("provenance", {})["deployment_name"] = chosen
    return Prepared(
        cluster_id,
        session["sessionId"],
        chosen,
        configuration,
        resolved,
        None,
        None,
        digest=_digest(cluster_id, resolved, _executable(configuration, None, chosen)),
    )


async def prepare(
    ctx: ToolContext,
    *,
    configuration: GuidePlanFields | None,
    from_deployment_id: str | None,
    cluster_id: str | None,
    deployment_name: str | None,
    model_cache_volume_id: str | None,
) -> Prepared:
    if (configuration is None) == (from_deployment_id is None):
        raise ToolError(
            "invalid_argument",
            "Provide exactly one of configuration or from_deployment_id.",
            field="configuration",
            hint="configuration takes the guide fields; from_deployment_id copies an existing deployment.",
        )
    if from_deployment_id:
        if model_cache_volume_id:
            raise ToolError(
                "invalid_argument", "A copy keeps the storage binding of its source.", field="model_cache_volume_id"
            )
        return await _from_deployment(ctx, from_deployment_id, cluster_id, deployment_name)
    if not cluster_id:
        raise ToolError(
            "invalid_argument",
            "cluster_id is required with configuration.",
            field="cluster_id",
            hint="See cluster_list.",
        )
    return await _from_configuration(ctx, configuration, cluster_id, deployment_name, model_cache_volume_id)  # type: ignore[arg-type]


def _run_body(prepared: Prepared, description: str | None, **plan: Any) -> dict[str, Any]:
    return {
        "configurations": [prepared.configuration],
        "provenance": {
            "cluster_session_id": prepared.session_id,
            "cluster_server_id": prepared.cluster_id,
            "created_via": CREATED_VIA,
            **({"description": description} if description is not None else {}),
        },
        **plan,
    }


async def freeze_plan(ctx: ToolContext, prepared: Prepared, description: str | None) -> dict[str, Any]:
    """Have Lens freeze and preflight the deployment; stores a plan but creates no run and no cluster resource."""
    return await ctx.api.request("POST", "/deployments/plans", body=_run_body(prepared, description), timeout=120)


async def launch(
    ctx: ToolContext,
    prepared: Prepared,
    description: str | None,
    *,
    plan_id: str | None = None,
    plan_digest: str | None = None,
    acknowledge_unknown: bool = False,
) -> dict[str, Any]:
    """Create the Lens deployment run; its id is the deployment id."""
    plan = (
        {"prepared_plan_id": plan_id, "expected_plan_digest": plan_digest, "acknowledge_unknown": acknowledge_unknown}
        if plan_id
        else {}
    )
    return await ctx.api.request(
        "POST", "/deployments/runs", body=_run_body(prepared, description, **plan), timeout=120
    )
