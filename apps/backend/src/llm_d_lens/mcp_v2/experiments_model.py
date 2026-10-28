"""Experiment inputs and the translation of workload presets into Lens benchmark specs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ToolError

ID_PATTERN = r"^[A-Za-z0-9_-]{1,40}$"
Percentile = Literal["p50", "p90", "p95", "p99"]
Preset = Literal["chat", "latency-sweep", "throughput-sweep", "shared-prefix", "long-input", "long-output"]


class SlaTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = Field(default=None, gt=0, description="Maximum time to first token in milliseconds.")
    ttft_percentile: Percentile = Field(default="p99", description="Percentile the ttft_ms target applies to.")
    tpot_ms: float | None = Field(default=None, gt=0, description="Maximum time per output token in milliseconds.")
    tpot_percentile: Percentile = Field(default="p99", description="Percentile the tpot_ms target applies to.")
    throughput_min_tps: float | None = Field(default=None, gt=0, description="Minimum output tokens per second.")
    success_rate_min_percent: float | None = Field(default=None, ge=0, le=100, description="Minimum success rate.")


class VariantSpec(BaseModel):
    """Same fields as plan_deployment (minus cluster_id); all optional so common_spec can supply defaults."""

    model_config = ConfigDict(extra="forbid")

    guide: str | None = Field(default=None, description="llm-d guide id from list_guides, e.g. 'optimized-baseline'.")
    model: str | None = Field(default=None, description="Hugging Face model id, e.g. 'Qwen/Qwen3-0.6B'.")
    accelerator: str | None = Field(default=None, description="'xpu' or 'gpu'. Inferred from the cluster if unique.")
    model_server: str | None = Field(default=None, description="'vllm' or 'sglang'. Defaults to vllm.")
    variant: str | None = Field(default=None, description="Guide variant. Defaults to 'base'.")
    replicas: int | None = Field(default=None, ge=1, le=64, description="Model-server replicas. Default 1.")
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=64, description="Accelerators per replica.")
    prefill_replicas: int | None = Field(default=None, ge=1, le=64, description="pd-disaggregation only.")
    prefill_tensor_parallel_size: int | None = Field(default=None, ge=1, le=64, description="pd-disaggregation only.")
    max_model_len: int | None = Field(default=None, ge=128, description="vLLM context length override.")
    image: str | None = Field(default=None, description="Override the model-server image.")


class ExperimentVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(pattern=ID_PATTERN, description="Unique within the experiment; used in result references.")
    label: str | None = Field(default=None, max_length=80, description="Human-readable name for result tables.")
    deployment_id: str | None = Field(
        default=None,
        pattern=r"^dep-",
        description="An existing deployment you own and that is ready. It is borrowed: loaded with benchmark "
        "traffic but never modified or deleted by the experiment.",
    )
    spec: VariantSpec | None = Field(
        default=None,
        description="A deployment the experiment creates itself (a temporary deployment, deleted when the variant "
        "finishes unless keep_deployments). Merged over common_spec.",
    )

    @model_validator(mode="after")
    def exactly_one_target(self) -> ExperimentVariant:
        if (self.deployment_id is None) == (self.spec is None):
            raise ValueError("set exactly one of deployment_id or spec")
        return self


class RateStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_rps: float = Field(gt=0, le=10000, description="Target requests per second (open loop).")
    duration_seconds: int = Field(ge=10, le=3600, description="Stage duration.")


class CustomWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: Literal["random", "shared_prefix"] = Field(
        default="random",
        description="random=fixed-length random prompts at closed-loop concurrency stages; shared_prefix=open-loop "
        "requests that reuse a few long system prompts (stresses prefix caching and prefix-aware routing).",
    )
    input_tokens: list[int] = Field(
        default_factory=lambda: [512], min_length=1, max_length=10, description="random: prompt lengths to sweep."
    )
    output_tokens: list[int] = Field(
        default_factory=lambda: [128], min_length=1, max_length=10, description="Generated lengths to sweep."
    )
    concurrency: list[int] = Field(
        default_factory=lambda: [1, 8],
        min_length=1,
        max_length=8,
        description="random: concurrency stages run at every (input, output) point.",
    )
    requests_per_stage: int | None = Field(
        default=None, ge=1, le=100000, description="random: requests per stage; default max(32, 4 x concurrency)."
    )
    num_groups: int = Field(default=32, ge=1, le=100000, description="shared_prefix: distinct system prompts.")
    prompts_per_group: int = Field(default=16, ge=1, le=10000, description="shared_prefix: requests per system prompt.")
    system_prompt_tokens: int = Field(default=2048, ge=1, le=131072, description="shared_prefix: shared prefix length.")
    question_tokens: int = Field(default=128, ge=1, le=131072, description="shared_prefix: per-request suffix length.")
    rate_stages: list[RateStage] = Field(
        default_factory=lambda: [
            RateStage(rate_rps=4, duration_seconds=60),
            RateStage(rate_rps=8, duration_seconds=60),
        ],
        min_length=1,
        max_length=20,
        description="shared_prefix: open-loop rate stages.",
    )

    @model_validator(mode="after")
    def bounded(self) -> CustomWorkload:
        if any(not 1 <= value <= 131072 for value in [*self.input_tokens, *self.output_tokens]):
            raise ValueError("token lengths must be between 1 and 131072")
        if any(not 1 <= value <= 1024 for value in self.concurrency):
            raise ValueError("concurrency values must be between 1 and 1024")
        if self.generator == "random" and len(self.input_tokens) * len(self.output_tokens) > 50:
            raise ValueError("at most 50 input x output points per workload")
        return self


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload_id: str | None = Field(default=None, pattern=ID_PATTERN, description="Defaults to the preset name.")
    preset: Preset | None = Field(
        default=None,
        description="chat=short smoke test (about 3 min); latency-sweep=concurrency 1..64; throughput-sweep="
        "concurrency 16..256 to saturation; shared-prefix=shared system prompts at 4 then 8 req/s; "
        "long-input=8k prompt, short output (prefill); long-output=short prompt, 2k output (decode).",
    )
    custom: CustomWorkload | None = Field(default=None, description="Alternative to preset.")
    sla: SlaTargets | None = Field(default=None, description="Overrides the experiment-level sla for this workload.")

    @model_validator(mode="after")
    def exactly_one_source(self) -> WorkloadSpec:
        if (self.preset is None) == (self.custom is None):
            raise ValueError("set exactly one of preset or custom")
        return self


def _stages(concurrency: list[int], requests: int | None) -> list[dict[str, int]]:
    return [{"concurrency": c, "num_requests": requests or min(max(32, 4 * c), 100000)} for c in concurrency]


def _matrix(points: list[tuple[int, int]], concurrency: list[int], requests: int | None, warmup: int):
    return {
        "matrix": [{"isl": isl, "osl": osl} for isl, osl in points],
        "concurrency_stages": _stages(concurrency, requests),
        "warmup_requests": warmup,
    }


def _shared_prefix(groups: int, per_group: int, system: int, question: int, output: int, stages: list[dict[str, Any]]):
    return {
        "shared_prefix": {
            "num_groups": groups,
            "num_prompts_per_group": per_group,
            "system_prompt_len": system,
            "question_len": question,
            "output_len": output,
            "stages": stages,
        }
    }


def _preset_benchmark(preset: str, warmup: int) -> dict[str, Any]:
    sweep = [1, 2, 4, 8, 16, 32, 64]
    if preset == "chat":
        return _matrix([(512, 128)], [1, 4], None, warmup)
    if preset == "latency-sweep":
        return _matrix([(1024, 128)], sweep, None, warmup)
    if preset == "throughput-sweep":
        return _matrix([(512, 256)], [16, 32, 64, 128, 256], None, warmup)
    if preset == "long-input":
        return _matrix([(8192, 128)], [1, 4, 16], None, warmup)
    if preset == "long-output":
        return _matrix([(128, 2048)], [1, 4, 16], None, warmup)
    return _shared_prefix(32, 16, 2048, 128, 256, [{"rate": 4, "duration": 60}, {"rate": 8, "duration": 60}])


def _custom_benchmark(custom: CustomWorkload, warmup: int) -> dict[str, Any]:
    if custom.generator == "shared_prefix":
        stages = [{"rate": s.rate_rps, "duration": s.duration_seconds} for s in custom.rate_stages]
        return _shared_prefix(
            custom.num_groups,
            custom.prompts_per_group,
            custom.system_prompt_tokens,
            custom.question_tokens,
            custom.output_tokens[0],
            stages,
        )
    points = [(i, o) for i in custom.input_tokens for o in custom.output_tokens]
    return _matrix(points, custom.concurrency, custom.requests_per_stage, warmup)


def resolve_workloads(
    workloads: list[WorkloadSpec], global_sla: SlaTargets | None, warmup: int
) -> list[dict[str, Any]]:
    """Plain-dict workloads: ``benchmark`` holds the Lens BenchmarkSpec fields, ``sla`` the targets (or None)."""
    resolved, seen = [], set()
    for index, workload in enumerate(workloads):
        workload_id = workload.workload_id or workload.preset or f"custom-{index + 1}"
        if workload_id in seen:
            raise ToolError(
                "invalid_argument",
                f"Duplicate workload_id '{workload_id}'.",
                field=f"workloads/{index}/workload_id",
                hint="Give each workload a unique workload_id.",
            )
        seen.add(workload_id)
        benchmark = (
            _preset_benchmark(workload.preset, warmup)
            if workload.preset
            else _custom_benchmark(workload.custom, warmup)
        )
        sla = workload.sla or global_sla
        resolved.append(
            {
                "workload_id": workload_id,
                "preset": workload.preset,
                "benchmark": benchmark,
                "sla": sla.model_dump(exclude_none=True) if sla else None,
            }
        )
    return resolved


def describe_workload(workload: dict[str, Any]) -> dict[str, Any]:
    """Short human-readable form of a resolved workload for plan output."""
    benchmark = workload["benchmark"]
    if "shared_prefix" in benchmark:
        return {"generator": "shared_prefix", **benchmark["shared_prefix"]}
    return {
        "generator": "random",
        "points": [f"{p['isl']}->{p['osl']} tokens" for p in benchmark["matrix"]],
        "concurrency_stages": benchmark["concurrency_stages"],
        "warmup_requests": benchmark["warmup_requests"],
    }
