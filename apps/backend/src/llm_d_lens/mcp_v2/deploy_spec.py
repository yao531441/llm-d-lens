"""DeploymentSpec: the intent-level description an agent passes to plan_deployment / create_deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .context import ToolContext
from .errors import ToolError

PD_GUIDE = "pd-disaggregation"
NAME_PATTERN = r"^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$"


class RoleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replicas: int = Field(default=1, ge=1, le=64, description="Pods in this role.")
    tensor_parallel_size: int = Field(default=1, ge=1, le=64, description="Accelerators per pod.")


class Topology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serving: RoleSpec | None = Field(default=None, description="Model servers of a non-PD guide.")
    prefill: RoleSpec | None = Field(default=None, description="pd-disaggregation only.")
    decode: RoleSpec | None = Field(default=None, description="pd-disaggregation only.")


class EngineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(default=None, ge=128, description="vLLM context length override.")


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(description="Cluster id from list_clusters.")
    model: str = Field(description="Hugging Face model id, e.g. 'Qwen/Qwen3-0.6B'.")
    guide: str | None = Field(
        default=None, description="llm-d guide id from list_guides, e.g. 'optimized-baseline'. Required to plan."
    )
    name: str | None = Field(default=None, pattern=NAME_PATTERN, description="Deployment name; generated when omitted.")
    description: str | None = Field(default=None, max_length=500)
    accelerator: str | None = Field(
        default=None, description="'xpu' or 'gpu'. Inferred when the cluster has exactly one family."
    )
    engine_kind: str | None = Field(default=None, description="'vllm' or 'sglang'. Defaults to vllm when offered.")
    variant: str | None = Field(default=None, description="Guide variant. Defaults to 'base'.")
    topology: Topology = Field(default_factory=Topology, description="Replicas and tensor parallelism per role.")
    engine: EngineSpec = Field(default_factory=EngineSpec)
    runtime_image: str | None = Field(default=None, description="Override the model-server image.")
    model_cache_volume_id: str | None = Field(
        default=None,
        description=(
            "Model-cache storage volume (Lens id or vol- id) that already holds the model. "
            "Chosen automatically when exactly one ready volume has it; otherwise call cache_model first."
        ),
    )


@dataclass(frozen=True)
class PlanParams:
    """The flat view of a spec that guide planning and rendering consume."""

    cluster_id: str
    guide: str
    model: str
    accelerator: str | None
    model_server: str | None
    variant: str | None
    replicas: int
    tensor_parallel_size: int
    prefill_replicas: int
    prefill_tensor_parallel_size: int
    max_model_len: int | None
    image: str | None
    name: str | None


def _invalid(message: str, field: str, hint: str) -> ToolError:
    return ToolError("unsupported_combination", message, field=field, hint=hint)


def plan_params(spec: DeploymentSpec) -> PlanParams:
    if not spec.guide:
        raise ToolError(
            "invalid_argument",
            "spec.guide is required to plan a deployment.",
            field="spec.guide",
            hint="Call list_guides, or plan_deployment(mode='recommend') to rank choices.",
        )
    topology = spec.topology
    if spec.guide == PD_GUIDE:
        if topology.serving:
            raise _invalid(
                "pd-disaggregation has prefill and decode roles, not serving.",
                "spec.topology.serving",
                "Use topology.prefill and topology.decode.",
            )
        prefill, decode = topology.prefill or RoleSpec(), topology.decode or RoleSpec()
    else:
        if topology.prefill or topology.decode:
            raise _invalid(
                f"Guide '{spec.guide}' has a single serving role.",
                "spec.topology",
                "Use topology.serving, or choose guide 'pd-disaggregation'.",
            )
        decode, prefill = topology.serving or RoleSpec(), RoleSpec(replicas=1, tensor_parallel_size=1)
    return PlanParams(
        cluster_id=spec.cluster_id,
        guide=spec.guide,
        model=spec.model,
        accelerator=spec.accelerator,
        model_server=spec.engine_kind,
        variant=spec.variant,
        replicas=decode.replicas,
        tensor_parallel_size=decode.tensor_parallel_size,
        prefill_replicas=prefill.replicas,
        prefill_tensor_parallel_size=prefill.tensor_parallel_size,
        max_model_len=spec.engine.max_model_len,
        image=spec.runtime_image,
        name=spec.name,
    )


def role_gpus(params: PlanParams) -> list[dict[str, Any]]:
    if params.guide == PD_GUIDE:
        return [
            {
                "role": "prefill",
                "replicas": params.prefill_replicas,
                "gpus_per_replica": params.prefill_tensor_parallel_size,
            },
            {"role": "decode", "replicas": params.replicas, "gpus_per_replica": params.tensor_parallel_size},
        ]
    return [{"role": "serving", "replicas": params.replicas, "gpus_per_replica": params.tensor_parallel_size}]


def gpus_total(params: PlanParams) -> int:
    return sum(role["replicas"] * role["gpus_per_replica"] for role in role_gpus(params))


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """RFC 7396 merge patch: null removes a key, objects merge, everything else replaces."""
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_spec(data: dict[str, Any], field: str = "spec") -> DeploymentSpec:
    try:
        return DeploymentSpec.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise ToolError(
            "invalid_argument",
            f"{field}.{location}: {first['msg']}" if location else str(first["msg"]),
            field=f"{field}.{location}" if location else field,
            hint="See the plan_deployment input schema for DeploymentSpec fields.",
        ) from error


def spec_from_deployment(ctx: ToolContext, deployment_id: str, patch: dict[str, Any] | None) -> DeploymentSpec:
    """The stored spec of one of the caller's own deployments with a merge patch applied."""
    row = ctx.ledger.get(deployment_id)
    if row is None or row.owner != ctx.owner or row.kind != "deployment":
        raise ToolError(
            "not_found",
            f"No deployment '{deployment_id}' created by you.",
            field="from_deployment_id",
            hint="Call list_resources(kinds=['deployment']) for ids; or pass a full spec.",
        )
    stored = row.meta.get("spec")
    if not stored:
        raise ToolError(
            "invalid_argument",
            f"Deployment '{deployment_id}' has no stored spec.",
            field="from_deployment_id",
            hint="Pass a full spec instead.",
        )
    base = {key: value for key, value in stored.items() if key != "name"}  # a clone needs a fresh name
    return parse_spec(_merge(base, patch or {}), "spec")
