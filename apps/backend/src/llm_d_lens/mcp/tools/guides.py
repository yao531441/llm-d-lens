"""llm-d guide discovery: which guides fit a cluster and which parameters the agent may set on each."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..cluster_info import cluster_readiness, fetch_overview, summarise_overview
from ..context import ToolContext
from ..errors import ToolError
from ..guide_fields import PD_GUIDE, GuidePlanFields
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
        "validations deployment_plan applies.",
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


def _field(name: str) -> dict[str, Any]:
    """Name, type and limits of one ``GuidePlanFields`` property, read from the model so they never drift."""
    schema = GuidePlanFields.model_json_schema()["properties"][name]
    kinds = [item for item in schema.get("anyOf", [schema]) if item.get("type") != "null"]
    detail = kinds[0] if kinds else schema
    entry: dict[str, Any] = {"name": name, "type": detail.get("type", "object")}
    entry.update({key: detail[key] for key in ("minimum", "maximum", "exclusiveMinimum") if key in detail})
    return entry


def _parameters(guide_id: str, supports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``configuration`` fields of deployment_plan that apply to this guide."""
    servers = sorted({server for item in supports for server in item["model_servers"]})
    variants = sorted({variant for item in supports for values in item["model_servers"].values() for variant in values})
    sizing = (
        ("prefillReplicas", "prefillTensorParallelSize", "decodeReplicas", "decodeTensorParallelSize")
        if guide_id == PD_GUIDE
        else ("replicas", "tensorParallelSize")
    )
    parameters: list[dict[str, Any]] = [
        {"name": "guide", "type": "string", "value": guide_id},
        {"name": "model", "type": "string", "required": True, "note": "Hugging Face model id."},
        {
            "name": "accelerator",
            "type": "enum",
            "values": [item["accelerator"] for item in supports],
            "default": "inferred from the cluster when it has exactly one family",
        },
        {
            "name": "modelServer",
            "type": "enum",
            "values": servers,
            "default": "vllm" if "vllm" in servers else None,
            "note": "Which servers exist depends on the accelerator; see supports.",
        },
        {
            "name": "guideVariant",
            "type": "enum",
            "values": variants,
            "default": "base",
            "note": "Lens does not describe the variants. A variant is a platform overlay folder of the llm-d guide "
            "for this accelerator and model server (for example 'rdma' or 'gke'); '.' is the folder's own "
            "kustomization and renders the same as 'base'. Omit it to use 'base'. What each variant changes is in "
            f"guides/{guide_id}/README.md of the llm-d repository at this response's `ref`, section "
            "'Configure the environment'.",
        },
    ]
    parameters += [{**_field(name), "default": 1} for name in sizing]
    parameters += [
        {**_field("maxModelLen"), "default": None, "applies_when": "modelServer is vllm"},
        {**_field("runtimeImage"), "default": "the guide's image"},
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
        hint="Call cluster_list.",
    )


@tool(
    "guide_list",
    "[R] List the llm-d deployment guides (e.g. optimized-baseline, pd-disaggregation) with the accelerator "
    "families, model servers and variants each supports, and whether each fits the cluster. Pass guide to see its "
    "typed parameters (replicas, tensor parallelism, max_model_len, prefill/decode roles), prerequisites and "
    "validations. Use the guide id in deployment_plan.\n"
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
                hint="Call guide_list without guide for the ids.",
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
