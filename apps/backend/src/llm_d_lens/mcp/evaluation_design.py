"""``ExperimentDesign`` and its expansion into a Lens Evaluation request. Pure functions."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_d_lens.evaluation.models import BenchmarkScenario, ComparisonCoreParameters

from .errors import ToolError
from .guide_fields import GuidePlanFields

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
BaselineType = Literal[
    "direct-vllm", "router-neutral", "load-only", "affinity-only", "optimized-baseline", "kubernetes-service"
]
MAX_PLANS, MAX_SCENARIOS, MAX_BASELINES = 50, 20, 7
MAX_NAME = 200
LARGE_EXPERIMENT_PLANS = 16


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(max_length=90, pattern=ID_PATTERN, description="Becomes the Lens plan id.")
    deployment_name: str | None = Field(None, min_length=1, max_length=120, description="Name of the deployment.")
    plan: GuidePlanFields | None = Field(None, description="Fields that override common for this variant.")


class Repetitions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(1, ge=1, le=10)
    mode: Literal["same_deployment", "fresh_deployment"] = Field(
        "same_deployment",
        description="same_deployment repeats each scenario on one deployment (later runs may see warm caches); "
        "fresh_deployment repeats the whole variant on a new deployment.",
    )


class ExperimentDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=MAX_NAME, description="Evaluation name.")
    cluster_id: str | None = Field(None, description="Cluster to run on; omit when there is exactly one.")
    common: GuidePlanFields | None = Field(None, description="Defaults for every variant.")
    model_cache_volume_id: str | None = Field(
        None,
        description="Model-cache volume to serve the model from when it is cached in more than one; "
        "omit to use the only volume holding it.",
    )
    variants: list[Variant] = Field(min_length=1, max_length=MAX_PLANS, description="Deployments to compare.")
    scenarios: list[BenchmarkScenario] = Field(
        min_length=1,
        max_length=MAX_SCENARIOS,
        description="Native Lens benchmark scenarios shared by every variant. "
        "Workload generators are mutually exclusive.",
    )
    repetitions: Repetitions = Field(default_factory=Repetitions)
    order: Literal["as_listed", "shuffled"] = Field("as_listed", description="shuffled permutes plan order with seed.")
    seed: int | None = Field(None, ge=0, le=2**31 - 1, description="Shuffle seed; derived from the design if omitted.")
    baseline_types: list[BaselineType] = Field(
        default_factory=list,
        max_length=MAX_BASELINES,
        description="Lens baselines run next to each variant; empty runs none.",
    )
    baseline_parameters: dict[BaselineType, ComparisonCoreParameters] = Field(
        default_factory=dict,
        description="Lens ComparisonCoreParameters keyed by baseline type. Its keys are snake_case request names "
        "(for example tensor_parallel_size, max_model_len), unlike the camelCase guide fields in common and variants; "
        "unknown keys are rejected with the allowed list.",
    )
    preserve_deployment: bool = Field(False, description="Keep each deployment after its last scenario.")
    compare_configurations: bool = Field(True, description="Ask Lens for pairwise comparison rows.")

    @field_validator("baseline_parameters", mode="before")
    @classmethod
    def validate_baseline_parameters(cls, parameters):
        if isinstance(parameters, dict):
            for baseline, values in parameters.items():
                _reject_unknown_keys(ComparisonCoreParameters, values, f"baseline_parameters/{baseline}")
        return parameters

    @field_validator("scenarios", mode="before")
    @classmethod
    def validate_scenarios(cls, scenarios):
        if not isinstance(scenarios, list):
            return scenarios
        for index, scenario in enumerate(scenarios):
            if isinstance(scenario, dict):
                _check_scenario(scenario, index)
        return scenarios


def scenario_payload(scenario: BenchmarkScenario) -> dict[str, Any]:
    """Keep omitted Lens defaults omitted when forwarding the native typed contract."""
    return scenario.model_dump(mode="json", exclude_unset=True)


@dataclass(frozen=True)
class PlanSpec:
    plan_id: str
    variant_id: str
    repetition: int
    deployment_name: str | None
    scenarios: list[dict[str, Any]]


def _limit(message: str, hint: str, field: str | None = None) -> ToolError:
    return ToolError("limit_exceeded", message, field=field, hint=hint)


def check_limits(design: ExperimentDesign) -> None:
    """Lens limits that would otherwise only fail after configuration artifacts were saved."""
    variant_ids = [variant.id for variant in design.variants]
    if len(set(variant_ids)) != len(variant_ids):
        raise ToolError("invalid_argument", "Variant ids must be unique.", field="variants", hint="Rename duplicates.")
    scenario_ids = [scenario.id for scenario in design.scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ToolError(
            "invalid_argument", "Scenario ids must be unique.", field="scenarios", hint="Rename duplicates."
        )
    for index, scenario in enumerate(design.scenarios):
        _check_scenario(scenario_payload(scenario), index)
    count, mode = design.repetitions.count, design.repetitions.mode
    if mode == "same_deployment" and len(design.scenarios) * count > MAX_SCENARIOS:
        raise _limit(
            f"{len(design.scenarios)} scenarios x {count} repetitions exceeds Lens' {MAX_SCENARIOS} scenarios "
            "per plan.",
            "Use repetitions.mode='fresh_deployment', reduce count, or split the experiment.",
            "repetitions",
        )
    if mode == "fresh_deployment" and len(design.variants) * count > MAX_PLANS:
        raise _limit(
            f"{len(design.variants)} variants x {count} repetitions exceeds Lens' {MAX_PLANS} plans per evaluation.",
            "Reduce count, use repetitions.mode='same_deployment', or split the experiment.",
            "repetitions",
        )
    if len(set(design.baseline_types)) != len(design.baseline_types):
        raise ToolError("invalid_argument", "baseline_types must be unique.", field="baseline_types")


def _nested_models(annotation: Any) -> list[type[BaseModel]]:
    """BaseModel classes inside an annotation such as ``list[Point] | None``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for argument in get_args(annotation) for model in _nested_models(argument)]


def _reject_unknown_keys(model: type[BaseModel], value: Any, path: str) -> None:
    """Lens' native models ignore unknown keys, so a typo would run with defaults without any sign of it."""
    if not isinstance(value, dict):
        return
    unknown = sorted(set(value) - set(model.model_fields))
    if unknown:
        raise ToolError(
            "invalid_argument",
            f"{path} has unknown key(s) {unknown}; Lens would ignore them and use defaults.",
            field=path,
            hint=f"Allowed keys: {sorted(model.model_fields)}.",
        )
    for name, info in model.model_fields.items():
        nested = _nested_models(info.annotation)
        child = value.get(name)
        if not nested or child is None:
            continue
        for index, item in enumerate(child if isinstance(child, list) else [child]):
            _reject_unknown_keys(nested[0], item, f"{path}/{name}" + (f"/{index}" if isinstance(child, list) else ""))


def _check_scenario(scenario: dict[str, Any], index: int) -> None:
    field = f"scenarios/{index}"
    _reject_unknown_keys(BenchmarkScenario, scenario, field)
    for required in ("id", "name"):
        if not isinstance(scenario.get(required), str) or not scenario[required]:
            raise ToolError("invalid_argument", f"Scenario {index} needs a string {required}.", field=field)
    if len(scenario["id"]) > 90:
        raise _limit(f"Scenario id '{scenario['id']}' is longer than 90 characters.", "Shorten it.", field)
    benchmark = scenario.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise ToolError("invalid_argument", "benchmark must be an object.", field=f"{field}/benchmark")
    shared = benchmark.get("shared_prefix", {})
    if shared is None:
        shared = {}
    if not isinstance(shared, dict):
        raise ToolError(
            "invalid_argument", "shared_prefix must be an object.", field=f"{field}/benchmark/shared_prefix"
        )
    sized = (
        ("matrix", benchmark.get("matrix"), 50),
        ("concurrency_stages", benchmark.get("concurrency_stages"), 20),
        ("shared_prefix.stages", shared.get("stages") if isinstance(shared, dict) else None, 50),
    )
    for name, value, limit in sized:
        if isinstance(value, list) and len(value) > limit:
            raise _limit(
                f"Scenario '{scenario['id']}' has {len(value)} {name} entries; Lens allows {limit}.",
                f"Split the sweep across several scenarios of at most {limit} points.",
                f"{field}/benchmark/{name}",
            )
    ranges = (("parallelism", 1, 32), ("wait_timeout_seconds", 1, 14400), ("warmup_requests", 0, 50))
    for name, low, high in ranges:
        value = benchmark.get(name)
        if value is not None and type(value) is not int:
            raise ToolError("invalid_argument", f"{name} must be an integer.", field=f"{field}/benchmark/{name}")
        if isinstance(value, (int, float)) and not low <= value <= high:
            raise _limit(
                f"Scenario '{scenario['id']}' {name}={value} is outside {low}..{high}.",
                f"Use a value between {low} and {high}.",
                f"{field}/benchmark/{name}",
            )
    for collection, limits in (
        ("matrix", (("isl", 1, 1_000_000), ("osl", 1, 1_000_000))),
        ("concurrency_stages", (("concurrency", 1, 4096), ("num_requests", 1, 100_000))),
    ):
        points = benchmark.get(collection, [])
        if not isinstance(points, list):
            raise ToolError(
                "invalid_argument", f"{collection} must be an array.", field=f"{field}/benchmark/{collection}"
            )
        for point_index, point in enumerate(points):
            for name, low, high in limits:
                value = point.get(name) if isinstance(point, dict) else None
                path = f"{field}/benchmark/{collection}/{point_index}/{name}"
                if type(value) is not int:
                    raise ToolError("invalid_argument", f"{name} must be an integer.", field=path)
                if not low <= value <= high:
                    raise _limit(
                        f"{name}={value} is outside {low}..{high}.", "Use a value in the supported range.", path
                    )
    generators = [name for name in ("matrix", "shared_prefix", "workload_yaml") if benchmark.get(name)]
    if len(generators) > 1:
        raise ToolError(
            "invalid_argument",
            f"Scenario '{scenario['id']}' sets {', '.join(generators)}; they are mutually exclusive.",
            field=f"{field}/benchmark",
            hint="Keep one workload generator per scenario.",
        )


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def resolve_seed(design: ExperimentDesign) -> int:
    """The given seed, else one derived from the design so planning and starting agree."""
    if design.seed is not None:
        return design.seed
    payload = design.model_dump(by_alias=True, exclude={"seed"}, mode="json")
    return int(hashlib.sha256(canonical(payload).encode()).hexdigest()[:8], 16) % 2**31


def _scenario_copy(scenario: dict[str, Any], repetition: int, count: int) -> dict[str, Any]:
    if count == 1:
        return dict(scenario)
    suffix = f" (r{repetition}/{count})"
    return {**scenario, "id": f"{scenario['id']}-r{repetition}", "name": scenario["name"][: 200 - len(suffix)] + suffix}


def expand(design: ExperimentDesign, seed: int) -> list[PlanSpec]:
    count, mode = design.repetitions.count, design.repetitions.mode
    plans: list[PlanSpec] = []
    if mode == "same_deployment":
        scenarios = [
            _scenario_copy(scenario_payload(scenario), repetition, count)
            for repetition in range(1, count + 1)
            for scenario in design.scenarios
        ]
        plans = [PlanSpec(v.id, v.id, 1, v.deployment_name, scenarios) for v in design.variants]
    else:
        for repetition in range(1, count + 1):
            for variant in design.variants:
                plan_id = variant.id if count == 1 else f"{variant.id}-r{repetition}"
                scenarios = [scenario_payload(scenario) for scenario in design.scenarios]
                plans.append(PlanSpec(plan_id, variant.id, repetition, variant.deployment_name, scenarios))
    if design.order == "shuffled":
        random.Random(seed).shuffle(plans)  # noqa: S311 - reproducible order, not security
    return plans


def description(plan: PlanSpec, design: ExperimentDesign, seed: int) -> str:
    total = design.repetitions.count if design.repetitions.mode == "fresh_deployment" else 1
    shown_seed = seed if design.order == "shuffled" else "none"
    return f"lens-mcp variant={plan.variant_id} rep={plan.repetition}/{total} order={design.order} seed={shown_seed}"


def benchmark_plans(
    design: ExperimentDesign, plans: list[PlanSpec], artifact_ids: dict[str, str], seed: int
) -> list[dict[str, Any]]:
    result = []
    for plan in plans:
        entry: dict[str, Any] = {
            "id": plan.plan_id,
            "configuration_artifact_id": artifact_ids[plan.variant_id],
            "deployment_description": description(plan, design, seed),
            "include_configuration": True,
            "include_baseline": bool(design.baseline_types),
            "baseline_types": list(design.baseline_types),
            "preserve_deployment": design.preserve_deployment,
            "scenarios": plan.scenarios,
        }
        if design.baseline_parameters:
            entry["baseline_parameters"] = {
                key: value.model_dump(mode="json", exclude_unset=True)
                for key, value in design.baseline_parameters.items()
            }
        if plan.deployment_name:
            entry["deployment_name"] = plan.deployment_name
        result.append(entry)
    return result


def evaluation_request(
    design: ExperimentDesign,
    plans: list[PlanSpec],
    artifact_ids: dict[str, str],
    session_id: str,
    seed: int,
    name: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "cluster_session_id": session_id,
        "benchmark_plans": benchmark_plans(design, plans, artifact_ids, seed),
        "compare_configurations": design.compare_configurations,
    }


def predicted_cases(design: ExperimentDesign, plans: list[PlanSpec]) -> list[dict[str, Any]]:
    """Cases in the order Lens will run them: guides, each followed by its same-pod baseline, then other baselines."""
    guides: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    for plan_index, plan in enumerate(plans, 1):
        for scenario_index, scenario in enumerate(plan.scenarios, 1):
            guide_id = f"guide-{plan_index}-{scenario_index}"
            guides.append({"id": guide_id, "kind": "guide", "plan": plan.plan_id, "scenario": scenario["id"]})
            for baseline_type in design.baseline_types:
                if baseline_type == "kubernetes-service":
                    key = f"k8s:{guide_id}"
                else:
                    key = f"{plan.plan_id}:{baseline_type}:{canonical(scenario.get('benchmark') or {})}"
                if key not in baselines:
                    baselines[key] = {
                        "id": f"baseline-{len(baselines) + 1}",
                        "kind": "baseline",
                        "plan": plan.plan_id,
                        "scenario": scenario["id"],
                        "baseline_type": baseline_type,
                        "after": guide_id if baseline_type == "kubernetes-service" else None,
                    }
    ordered: list[dict[str, Any]] = []
    for guide in guides:
        ordered.append(guide)
        ordered.extend(item for item in baselines.values() if item["after"] == guide["id"])
    ordered.extend(item for item in baselines.values() if item["after"] is None)
    return [{key: value for key, value in item.items() if key != "after"} for item in ordered]
