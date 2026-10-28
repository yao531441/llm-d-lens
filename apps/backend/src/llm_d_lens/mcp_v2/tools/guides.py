"""llm-d guide discovery: which guides fit a cluster and which parameters the agent may set on each."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..cluster_info import cluster_readiness, fetch_overview, summarise_overview
from ..context import ToolContext
from ..deploy_spec import NAME_PATTERN, PD_GUIDE, EngineSpec, RoleSpec
from ..errors import ToolError
from ..lens_api import LensApiError
from ..registry import tool, upstream_error


class ListGuidesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str | None = Field(
        default=None,
        description="Resolve the guide catalog against this cluster's pinned llm-d ref and report how each guide "
        "fits the cluster. Optional when Lens manages exactly one cluster.",
    )
    accelerator: str | None = Field(
        default=None, description="Only guides that support this accelerator family, e.g. 'xpu' or 'gpu'."
    )
    guide: str | None = Field(
        default=None,
        description="Show one guide in detail: the typed parameters you may set on it, prerequisites and the "
        "validations plan_deployment applies.",
    )


def _roles(guide_id: str) -> list[str]:
    return ["prefill", "decode"] if guide_id == PD_GUIDE else ["serving"]


def _supports(guide: dict[str, Any], accelerator: str | None) -> list[dict[str, Any]]:
    return [
        {
            "accelerator": item["id"],
            "model_servers": {server["id"]: server.get("variants") or [] for server in item.get("modelServers") or []},
        }
        for item in guide.get("accelerators") or []
        if not accelerator or item["id"] == accelerator
    ]


def _limits(model: type[BaseModel], name: str) -> dict[str, Any]:
    schema = model.model_json_schema()["properties"][name]
    properties = next((item for item in schema.get("anyOf", []) if "minimum" in item), schema)
    return {key: properties[key] for key in ("minimum", "maximum") if key in properties}


def _parameters(guide_id: str, supports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    servers = sorted({server for item in supports for server in item["model_servers"]})
    variants = sorted({variant for item in supports for values in item["model_servers"].values() for variant in values})
    parameters: list[dict[str, Any]] = [
        {
            "name": "accelerator",
            "path": "spec.accelerator",
            "type": "enum",
            "values": [item["accelerator"] for item in supports],
            "default": "inferred from the cluster when it has exactly one family",
        },
        {
            "name": "model_server",
            "path": "spec.engine_kind",
            "type": "enum",
            "values": servers,
            "default": "vllm" if "vllm" in servers else None,
            "note": "Which servers exist depends on the accelerator; see supports.",
        },
        {"name": "variant", "path": "spec.variant", "type": "enum", "values": variants, "default": "base"},
    ]
    for role in _roles(guide_id):
        for field, label in (("replicas", "replicas"), ("tensor_parallel_size", "tensor_parallel_size")):
            parameters.append(
                {
                    "name": f"{role}_{label}" if guide_id == PD_GUIDE else label,
                    "path": f"spec.topology.{role}.{field}",
                    "type": "integer",
                    "default": 1,
                    **_limits(RoleSpec, field),
                    "note": "Pods in this role." if field == "replicas" else "Accelerators per pod.",
                }
            )
    parameters += [
        {
            "name": "max_model_len",
            "path": "spec.engine.max_model_len",
            "type": "integer",
            "default": None,
            "applies_when": "model_server is vllm",
            **_limits(EngineSpec, "max_model_len"),
            "note": "Context length override; get_model reports the model's limit and per-TP memory fit.",
        },
        {"name": "image", "path": "spec.runtime_image", "type": "string", "default": "the guide's image"},
        {"name": "name", "path": "spec.name", "type": "string", "pattern": NAME_PATTERN, "default": "generated"},
    ]
    return parameters


_VALIDATIONS = [
    "The guide exists in the catalog pinned for the cluster.",
    "The guide supports the accelerator family, the model server and the variant.",
    "Guide planning verifies the rendered configuration and returns validation errors for unfit sizing.",
    "Total accelerators (replicas x tensor_parallel_size over all roles) must be free on the cluster.",
]


async def _catalog(ctx: ToolContext, cluster_id: str) -> dict[str, Any]:
    try:
        return await ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": cluster_id})
    except LensApiError as error:
        raise upstream_error(error) from error


async def _resolve_cluster(ctx: ToolContext, cluster_id: str | None) -> str:
    if cluster_id:
        return cluster_id
    data = await ctx.api.request("GET", "/cluster/clusters")
    ids = [item["id"] for item in data.get("items") or []]
    if len(ids) == 1:
        return ids[0]
    raise ToolError(
        "invalid_argument",
        "The guide catalog is pinned per cluster, so cluster_id is required."
        + (f" Clusters: {', '.join(ids)}." if ids else ""),
        field="cluster_id",
        hint="Call list_clusters.",
    )


@tool(
    "list_guides",
    "[R] List the llm-d deployment guides (e.g. optimized-baseline, pd-disaggregation) with the accelerator "
    "families, model servers and variants each supports, and whether each fits the cluster. Pass guide to see its "
    "typed parameters (replicas, tensor parallelism, max_model_len, prefill/decode roles), prerequisites and "
    "validations. Use the guide id in plan_deployment.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    ListGuidesArgs,
    read_only=True,
    open_world=True,
)
async def list_guides(ctx: ToolContext, args: ListGuidesArgs) -> dict[str, Any]:
    cluster_id = await _resolve_cluster(ctx, args.cluster_id)
    catalog = await _catalog(ctx, cluster_id)
    try:
        overview = await fetch_overview(ctx, cluster_id)
    except LensApiError:
        overview = None
    summary = summarise_overview(cluster_id, overview) if overview else None
    families = {item["family"] for item in summary["accelerators"]} if summary else set()

    if args.guide:
        guide = next((item for item in catalog.get("guides") or [] if item["id"] == args.guide), None)
        if guide is None:
            raise ToolError(
                "not_found",
                f"Guide '{args.guide}' is not in the catalog.",
                field="guide",
                hint="Call list_guides without guide for the ids.",
            )
        supports = _supports(guide, args.accelerator)
        detail: dict[str, Any] = {
            "guide": guide["id"],
            "label": guide.get("label"),
            "roles": _roles(guide["id"]),
            "supports": supports,
            "parameters": _parameters(guide["id"], supports),
            "validations": _VALIDATIONS,
        }
        prerequisites: list[dict[str, Any]] = [
            {"component": "gateway-api-crds"},
            {"component": "inference-extension-crds"},
            {"component": "accelerator device plugin"},
            {"component": "hf-token-secret", "when": "the model is gated"},
        ]
        if overview:
            readiness = await cluster_readiness(ctx, cluster_id, overview)
            status = {item["component"]: item["status"] for item in readiness["items"]}
            plugins = [value for name, value in status.items() if name.startswith("device-plugin:")]
            status["accelerator device plugin"] = (
                "unknown" if not plugins else "ready" if all(value == "ready" for value in plugins) else "degraded"
            )
            for item in prerequisites:
                item["status"] = status.get(item["component"], "unknown")
            detail["cluster_ready"] = readiness["overall"]
        detail["prerequisites"] = prerequisites
        return {"cluster_id": cluster_id, "ref": catalog.get("ref"), "guide_detail": detail}

    guides = []
    for guide in catalog.get("guides") or []:
        supports = _supports(guide, args.accelerator)
        if not supports:
            continue
        entry: dict[str, Any] = {
            "guide": guide["id"],
            "label": guide.get("label"),
            "roles": _roles(guide["id"]),
            "supports": supports,
        }
        if summary:
            offered = [item["accelerator"] for item in supports]
            entry["cluster_fit"] = {
                "fits": bool(families & set(offered)),
                "cluster_accelerators": sorted(families),
                "accelerators_free": summary["accelerators_free"],
            }
        guides.append(entry)
    return {"cluster_id": cluster_id, "ref": catalog.get("ref"), "guides": guides}
