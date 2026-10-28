"""``ExperimentDesign`` and its expansion into a Lens Evaluation request (design 7.2 and 7.3). Pure functions."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    variants: list[Variant] = Field(min_length=1, max_length=MAX_PLANS, description="Deployments to compare.")
    scenarios: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_SCENARIOS,
        description="Lens BenchmarkScenario objects {id, name, benchmark, sla_targets}, passed verbatim and shared "
        "by every variant.",
    )
    repetitions: Repetitions = Field(default_factory=Repetitions)
    order: Literal["as_listed", "shuffled"] = Field("as_listed", description="shuffled permutes plan order with seed.")
    seed: int | None = Field(None, ge=0, le=2**31 - 1, description="Shuffle seed; derived from the design if omitted.")
    baseline_types: list[BaselineType] = Field(
        default_factory=list,
        max_length=MAX_BASELINES,
        description="Lens baselines run next to each variant; empty runs none.",
    )
    baseline_parameters: dict[BaselineType, dict[str, Any]] = Field(
        default_factory=dict, description="Lens ComparisonCoreParameters keyed by baseline type."
    )
    preserve_deployment: bool = Field(False, description="Keep each deployment after its last scenario.")
    compare_configurations: bool = Field(True, description="Ask Lens for pairwise comparison rows.")


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
    scenario_ids = [str(scenario.get("id")) for scenario in design.scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ToolError(
            "invalid_argument", "Scenario ids must be unique.", field="scenarios", hint="Rename duplicates."
        )
    for index, scenario in enumerate(design.scenarios):
        _check_scenario(scenario, index)
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


def _check_scenario(scenario: dict[str, Any], index: int) -> None:
    field = f"scenarios/{index}"
    for required in ("id", "name"):
        if not isinstance(scenario.get(required), str) or not scenario[required]:
            raise ToolError("invalid_argument", f"Scenario {index} needs a string {required}.", field=field)
    if len(scenario["id"]) > 90:
        raise _limit(f"Scenario id '{scenario['id']}' is longer than 90 characters.", "Shorten it.", field)
    benchmark = scenario.get("benchmark") or {}
    shared = benchmark.get("shared_prefix") or {}
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
        if isinstance(value, (int, float)) and not low <= value <= high:
            raise _limit(
                f"Scenario '{scenario['id']}' {name}={value} is outside {low}..{high}.",
                f"Use a value between {low} and {high}.",
                f"{field}/benchmark/{name}",
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
            _scenario_copy(scenario, repetition, count)
            for repetition in range(1, count + 1)
            for scenario in design.scenarios
        ]
        plans = [PlanSpec(v.id, v.id, 1, v.deployment_name, scenarios) for v in design.variants]
    else:
        for repetition in range(1, count + 1):
            for variant in design.variants:
                plan_id = variant.id if count == 1 else f"{variant.id}-r{repetition}"
                scenarios = [dict(scenario) for scenario in design.scenarios]
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
            entry["baseline_parameters"] = design.baseline_parameters
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


def client_name(name: str, client_request_id: str | None) -> str:
    if not client_request_id:
        return name
    suffix = f" [req:{client_request_id}]"
    return name[: MAX_NAME - len(suffix)] + suffix
