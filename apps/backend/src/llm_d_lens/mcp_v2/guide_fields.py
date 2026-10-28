"""``GuidePlanFields``: the guide-planning request fields shared by experiment variants (design 5.2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import ToolError

PD_GUIDE = "pd-disaggregation"
PD_FIELDS = ("prefillReplicas", "prefillTensorParallelSize", "decodeReplicas", "decodeTensorParallelSize")
RUNTIME_FIELDS = ("maxModelLen", "maxNumSeqs", "gpuMemoryUtilization", "blockSize", "maxNumBatchedTokens")


class GuidePlanFields(BaseModel):
    """Properties of the Lens guide-planning request; only the listed defaults and the PD aliases are applied."""

    model_config = ConfigDict(extra="forbid")

    guide: str | None = Field(None, description="Catalog guide id, for example pd-disaggregation, optimized-baseline.")
    accelerator: str | None = Field(None, description="Catalog accelerator id; omit when the cluster has one family.")
    model_server: str | None = Field(None, alias="modelServer", description="Default vllm.")
    guide_variant: str | None = Field(None, alias="guideVariant", description="A catalog variant of the model server.")
    model: str | None = Field(None, description="Hugging Face model id.")
    replicas: int | None = Field(
        None, ge=0, description="Serving replicas; for pd-disaggregation an alias of decodeReplicas."
    )
    tensor_parallel_size: int | None = Field(
        None,
        alias="tensorParallelSize",
        ge=0,
        description="Tensor parallel size; for pd-disaggregation an alias of decodeTensorParallelSize.",
    )
    prefill_replicas: int | None = Field(None, alias="prefillReplicas", ge=0, description="pd-disaggregation only.")
    prefill_tensor_parallel_size: int | None = Field(
        None, alias="prefillTensorParallelSize", ge=0, description="pd-disaggregation only."
    )
    decode_replicas: int | None = Field(None, alias="decodeReplicas", ge=0, description="pd-disaggregation only.")
    decode_tensor_parallel_size: int | None = Field(
        None, alias="decodeTensorParallelSize", ge=0, description="pd-disaggregation only."
    )
    max_model_len: int | None = Field(None, alias="maxModelLen", ge=1, description="vLLM max-model-len.")
    max_num_seqs: int | None = Field(None, alias="maxNumSeqs", ge=1, description="vLLM max-num-seqs.")
    gpu_memory_utilization: float | None = Field(
        None, alias="gpuMemoryUtilization", gt=0, le=1, description="vLLM gpu-memory-utilization."
    )
    block_size: int | None = Field(None, alias="blockSize", ge=1, description="vLLM block-size.")
    max_num_batched_tokens: int | None = Field(
        None, alias="maxNumBatchedTokens", ge=1, description="vLLM max-num-batched-tokens."
    )
    custom_parameters: list[dict[str, Any]] | None = Field(
        None,
        alias="customParameters",
        description="Runtime overrides passed to Lens unchanged: [{target: prefill|decode|both, "
        "kind: argument|environment, name, value}].",
    )
    runtime_image: str | None = Field(None, alias="runtimeImage", description="Default: resolved by Lens.")
    model_source: str | None = Field(None, alias="modelSource", description="Model location, passed through.")
    model_path: str | None = Field(None, alias="modelPath", description="Model location, passed through.")
    model_pvc_claim_name: str | None = Field(
        None, alias="modelPvcClaimName", description="Model location, passed through."
    )
    guide_settings: dict[str, Any] | None = Field(
        None, alias="guideSettings", description="rdmaNicCount, cacheCpuGiB, routerValues."
    )


def merge_fields(common: GuidePlanFields | None, plan: GuidePlanFields | None) -> dict[str, Any]:
    """Variant fields override ``common``; keys are the Lens request names."""
    merged: dict[str, Any] = {}
    for part in (common, plan):
        if part is not None:
            merged.update(part.model_dump(by_alias=True, exclude_none=True))
    return merged


def _conflict(alias: str, target: str, variant_id: str) -> ToolError:
    return ToolError(
        "field_conflict",
        f"Variant '{variant_id}' sets both {alias} and {target} with different values.",
        field=f"variants/{variant_id}/plan/{alias}",
        hint=f"For {PD_GUIDE}, {alias} is an alias of {target}; set only one of them.",
    )


def normalize_fields(fields: dict[str, Any], variant_id: str) -> dict[str, Any]:
    """Validate guide/model, apply the PD aliasing and fill the replicas/TP values render needs."""
    result = dict(fields)
    for required in ("guide", "model"):
        if not result.get(required):
            raise ToolError(
                "invalid_argument",
                f"Variant '{variant_id}' has no {required}.",
                field=f"variants/{variant_id}/plan/{required}",
                hint=f"Set {required} in common or in the variant's plan.",
            )
    is_pd = result["guide"] == PD_GUIDE
    if not is_pd:
        misplaced = [name for name in PD_FIELDS if name in result]
        if misplaced:
            raise ToolError(
                "field_not_applicable",
                f"{', '.join(misplaced)} only apply to {PD_GUIDE}; the guide is '{result['guide']}'.",
                field=f"variants/{variant_id}/plan/{misplaced[0]}",
                hint="Use replicas and tensorParallelSize for non-PD guides.",
            )
        result.setdefault("replicas", 1)
        result.setdefault("tensorParallelSize", 1)
        return result
    for alias, target in (("replicas", "decodeReplicas"), ("tensorParallelSize", "decodeTensorParallelSize")):
        if alias in result and target in result and result[alias] != result[target]:
            raise _conflict(alias, target, variant_id)
        value = result.get(target, result.get(alias))
        if value is not None:
            result[target] = result[alias] = value
    missing = [name for name in PD_FIELDS if result.get(name) is None]
    if missing:
        raise ToolError(
            "invalid_argument",
            f"Variant '{variant_id}' needs {', '.join(missing)} for {PD_GUIDE}.",
            field=f"variants/{variant_id}/plan/{missing[0]}",
            hint="Set all four PD role fields (replicas and tensor parallel size for prefill and decode).",
        )
    return result


def plan_request(
    fields: dict[str, Any], *, accelerator: str, model_server: str, guide_variant: str, session_id: str
) -> dict[str, Any]:
    """Body of ``POST /api/guide-planning/plan``."""
    request = {key: value for key, value in fields.items() if value is not None and key != "storageVolumeId"}
    request.update(
        accelerator=accelerator,
        modelServer=model_server,
        guideVariant=guide_variant,
        clusterSessionId=session_id,
        environment={"mode": "current", "kubernetesMode": "required"},
    )
    request.setdefault("source", {"mode": "official"})
    request.setdefault("guideSettings", {})
    if model_server != "vllm":
        for name in RUNTIME_FIELDS:
            request.pop(name, None)
    return request


def candidate_config(
    fields: dict[str, Any], *, cluster: dict[str, Any], model_server: str, guide_variant: str, image: str
) -> dict[str, Any]:
    """The render ``candidate_config`` mirroring the planned topology; render requires explicit replicas and TP."""
    is_pd = fields["guide"] == PD_GUIDE
    vllm = model_server == "vllm"

    def role(replicas_key: str, tp_key: str) -> dict[str, Any]:
        component: dict[str, Any] = {"replicas": fields[replicas_key], "tensor_parallel_size": fields[tp_key]}
        if vllm and fields.get("maxModelLen"):
            component["max_model_len"] = fields["maxModelLen"]
        if vllm and fields.get("maxNumSeqs"):
            component["max_num_seqs"] = fields["maxNumSeqs"]
        return component

    decode = role("decodeReplicas", "decodeTensorParallelSize") if is_pd else role("replicas", "tensorParallelSize")
    runtime: dict[str, Any] = {
        "image": image,
        "imageMode": "use-upstream-image",
        "buildSourceUrl": "",
        "modelServer": model_server,
        "resourceBasis": "available",
        "modelPvcClaimName": fields.get("modelPvcClaimName") or "",
        "modelSource": fields.get("modelSource") or "huggingface",
        "mountPath": "",
    }
    if fields.get("storageVolumeId"):
        runtime["storageVolumeId"] = fields["storageVolumeId"]
    return {
        "schema_version": "1.0",
        "type": "pd" if is_pd else "baseline",
        "guide_settings": fields.get("guideSettings") or {},
        "candidate_source": {"name": "manual"},
        "target": {"model": fields["model"], "cluster_id": cluster["id"], "cluster_name": cluster["name"]},
        "serving": dict(decode),
        "prefill": role("prefillReplicas", "prefillTensorParallelSize") if is_pd else None,
        "decode": decode,
        "custom_parameters": fields.get("customParameters") or [],
        "guide_variant": guide_variant,
        "runtime": runtime,
    }


def gpu_count(fields: dict[str, Any]) -> int:
    if fields["guide"] == PD_GUIDE:
        return int(
            fields["prefillReplicas"] * fields["prefillTensorParallelSize"]
            + fields["decodeReplicas"] * fields["decodeTensorParallelSize"]
        )
    return int(fields["replicas"] * fields["tensorParallelSize"])
