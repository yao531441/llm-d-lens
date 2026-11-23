"""Simulation profile: replay a trace dataset against one of the caller's own deployments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..errors import ToolError
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..path_ids import check_id
from ..registry import tool
from ..simulation_target import resolve_target
from ..simulation_view import (
    ACTIVE,
    delete_simulation,
    estimate_replay,
    fetch_task,
    get_simulation_entry,
    list_simulations,
    public_status,
    response_code_diagnostics,
    shape_artifacts,
    shape_results,
    task_summary,
    wait_for_task,
)

PROFILE = "simulation"
TIMELINE_BINS = 200


class ListTraceDatasetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: list[Literal["estimate"]] = Field(
        default_factory=list,
        description="'estimate' adds the request count and peak request rate a replay of `dataset` would produce, "
        "so no trial run is needed before simulation_start.",
    )
    dataset: str | None = Field(None, description="Dataset name from this list; required with include=['estimate'].")
    duration_seconds: int = Field(60, ge=5, le=7200, description="Replay length used for the estimate.")
    scale_factor: float = Field(1.0, gt=0, le=50, description="Replay speed-up used for the estimate.")


class PrepareTraceDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="Dataset name from trace_dataset_list.")
    force: bool = Field(False, description="Download again even if the file is already present.")


class StartSimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(
        description="A deployment you created. Targets owned by someone else are refused, and model services "
        "are not supported."
    )
    dataset: str = Field(description="Downloaded dataset name from trace_dataset_list; it fixes the scenario.")
    duration_seconds: int = Field(60, ge=5, le=7200, description="How long the replay runs.")
    scale_factor: float = Field(
        1.0, gt=0, le=50, description="Speed-up of the trace; only datasets whose format supports it accept != 1."
    )
    backend: str | None = Field(None, description="Replay backend; default is the dataset's preferred one.")
    name: str | None = Field(None, max_length=120, description="Display name; generated when omitted.")


class GetSimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(description="Simulation id from simulation_start or resource_list.")
    include: list[Literal["results", "response_code_issues", "artifacts"]] = Field(
        default_factory=lambda: ["results"],
        description="results = summary metrics; response_code_issues = non-2xx codes with likely causes and samples; "
        "artifacts = report file names.",
    )
    wait_seconds: int = Field(0, ge=0, le=25, description="Wait up to this long for the simulation to settle.")


class CancelSimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(description="Simulation id from simulation_start or resource_list.")


def _scale_formats(backend: dict[str, Any]) -> list[str]:
    formats = (backend.get("capabilities") or {}).get("trace_format_capabilities") or {}
    return [name for name, caps in formats.items() if (caps or {}).get("supports_scale_factor")]


@tool(
    "trace_dataset_list",
    "[R] List the trace datasets, scenarios and replay backends a simulation can use, with whether each dataset is "
    "downloaded. include=['estimate'] with a dataset gives the request count and peak rate of a replay, replacing a "
    "dry run before simulation_start. To fetch a missing dataset use trace_dataset_prepare.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    ListTraceDatasetsInput,
    read_only=True,
    profile=PROFILE,
)
async def list_trace_datasets(ctx: ToolContext, args: ListTraceDatasetsInput) -> dict[str, Any]:
    if "estimate" in args.include and not args.dataset:
        raise ToolError("invalid_argument", "include=['estimate'] needs a dataset.", field="dataset")
    catalog = await ctx.api.request("GET", "/simulation/trace-datasets")
    scenarios = await ctx.api.request("GET", "/simulation/scenarios")
    backends = await ctx.api.request("GET", "/simulation/backends")
    result: dict[str, Any] = {
        "datasets": [
            {
                "name": item["name"],
                "scenario": item.get("scenario"),
                "trace_format": item.get("trace_format"),
                "downloaded": bool(item.get("downloaded")),
                "size_bytes": item.get("size_bytes"),
                "license": item.get("license"),
                "supported_backends": item.get("supported_backends"),
            }
            for item in catalog.get("datasets") or []
        ],
        "scenarios": [
            {"name": item["name"], "description": item.get("description"), "backends": item.get("supported_backends")}
            for item in scenarios.get("scenarios") or []
        ],
        "backends": [
            {
                "name": item["name"],
                "available": bool(item.get("available")),
                "trace_formats": (item.get("capabilities") or {}).get("trace_formats"),
                "scale_factor_formats": _scale_formats(item),
            }
            for item in backends.get("backends") or []
        ],
    }
    if "estimate" in args.include:
        entry = next((item for item in result["datasets"] if item["name"] == args.dataset), None)
        if entry is None:
            raise ToolError("not_found", f"No dataset '{args.dataset}'.", field="dataset")
        if not entry["downloaded"]:
            result["estimate"] = None
            result["next"] = f"Run trace_dataset_prepare for '{args.dataset}' before estimating."
        else:
            timeline = await ctx.api.request(
                "GET", f"/simulation/trace-datasets/{args.dataset}/timeline", params={"bins": TIMELINE_BINS}
            )
            result["estimate"] = estimate_replay(timeline, args.duration_seconds, args.scale_factor)
    return result


async def _dataset(ctx: ToolContext, name: str) -> dict[str, Any]:
    catalog = await ctx.api.request("GET", "/simulation/trace-datasets")
    entry = next((item for item in catalog.get("datasets") or [] if item.get("name") == name), None)
    if entry is None:
        raise ToolError("not_found", f"No dataset '{name}'.", field="dataset", hint="Call trace_dataset_list.")
    return entry


@tool(
    "trace_dataset_prepare",
    "[W] Download and validate a registered trace dataset so it can be replayed. Returns at once when the dataset is "
    "already present. Find names with trace_dataset_list; start the replay with simulation_start.\n"
    "Side effects: downloads a file onto the Lens host; resources: disk and outbound bandwidth; "
    "duration: synchronous, up to a few minutes for large traces; reversible: yes (the file can be re-downloaded).",
    PrepareTraceDatasetInput,
    read_only=False,
    idempotent=True,
    open_world=True,
    profile=PROFILE,
)
async def prepare_trace_dataset(ctx: ToolContext, args: PrepareTraceDatasetInput) -> dict[str, Any]:
    entry = await _dataset(ctx, args.dataset)
    if entry.get("downloaded") and not args.force:
        return {"dataset": args.dataset, "status": "ready", "size_bytes": entry.get("size_bytes"), "downloaded": False}
    try:
        await ctx.api.request(
            "POST",
            "/simulation/trace-datasets/download",
            body={"dataset": args.dataset, "force": args.force},
            timeout=600,
        )
    except LensApiError as error:
        if error.status == 429:
            raise ToolError(
                "conflict_state", "Lens is already running the maximum number of downloads.", hint="Retry shortly."
            ) from error
        raise
    entry = await _dataset(ctx, args.dataset)
    status = "ready" if entry.get("downloaded") else "failed"
    return {
        "dataset": args.dataset,
        "status": status,
        "size_bytes": entry.get("size_bytes"),
        "downloaded": True,
        "next": "Call simulation_start." if status == "ready" else "Retry trace_dataset_prepare; check the Lens logs.",
    }


async def _pick_backend(
    ctx: ToolContext, dataset: dict[str, Any], requested: str | None, scale: float
) -> tuple[str, bool]:
    """Return the backend name and whether it accepts a source-time range for this dataset's format."""
    supported = dataset.get("supported_backends") or []
    name = requested or ("trace-replayer" if "trace-replayer" in supported else (supported or [None])[0])
    if not name or name not in supported:
        raise ToolError(
            "unsupported_combination",
            f"Backend '{name}' cannot replay '{dataset['name']}' (supported: {', '.join(supported) or 'none'}).",
            field="backend",
        )
    backends = await ctx.api.request("GET", "/simulation/backends")
    backend = next((item for item in backends.get("backends") or [] if item.get("name") == name), None)
    if backend is None or not backend.get("available"):
        raise ToolError("conflict_state", f"Backend '{name}' is not available on this Lens.", field="backend")
    caps = ((backend.get("capabilities") or {}).get("trace_format_capabilities") or {}).get(
        dataset.get("trace_format"), {}
    )
    if scale != 1.0 and not caps.get("supports_scale_factor"):
        raise ToolError(
            "unsupported_combination",
            f"Backend '{name}' cannot scale the '{dataset.get('trace_format')}' trace; use scale_factor=1.",
            field="scale_factor",
        )
    return name, bool(caps.get("supports_trace_range"))


async def _ensure_target_idle(ctx: ToolContext, execution_id: str) -> None:
    page = await ctx.api.request("GET", "/simulation/tasks", params={"limit": 100})
    for task in page.get("tasks") or []:
        if task.get("endpoint_deployment_execution_id") == execution_id and task.get("status") in ACTIVE:
            raise ToolError(
                "conflict_state",
                f"Simulation {task['id']} is already replaying against this deployment.",
                field="target_id",
                hint="Wait for it with simulation_get, or stop it with simulation_cancel.",
            )


@tool(
    "simulation_start",
    "[S] Replay a trace dataset against a deployment you created and return its id; poll simulation_get. It "
    "loads the target for the whole duration, so only your own deployments are accepted and only one replay may run "
    "per deployment. For comparing configurations under controlled load use the experiment tools instead.\n"
    "Side effects: sends request traffic to the target; resources: target accelerators and a Lens replay task; "
    "duration: asynchronous, as long as duration_seconds; reversible: stop it with simulation_cancel.",
    StartSimulationInput,
    read_only=False,
    profile=PROFILE,
)
async def start_simulation(ctx: ToolContext, args: StartSimulationInput) -> dict[str, Any]:
    target = await resolve_target(ctx, args.target_id)
    dataset = await _dataset(ctx, args.dataset)
    if not dataset.get("downloaded"):
        raise ToolError(
            "conflict_state",
            f"Dataset '{args.dataset}' is not downloaded.",
            field="dataset",
            hint="Call trace_dataset_prepare first.",
        )
    backend, supports_range = await _pick_backend(ctx, dataset, args.backend, args.scale_factor)
    await _ensure_target_idle(ctx, target.execution_id)
    name = args.name or f"{dataset['name']}-{target.name or target.deployment_id}"[:120]
    body: dict[str, Any] = {
        **target.endpoint_fields,
        "name": name,
        "scenario": dataset["scenario"],
        "backend": backend,
        "trace_dataset": dataset["name"],
        "trace_path": dataset.get("public_dataset") or dataset["path"],
        "duration_seconds": args.duration_seconds,
    }
    if supports_range:
        body.update(trace_start_seconds=0, trace_end_seconds=args.duration_seconds * args.scale_factor)
    if args.scale_factor != 1.0:
        body["scale_factor"] = args.scale_factor
    created = await ctx.api.request("POST", "/simulation/tasks", body=body, timeout=120)
    return {
        "simulation_id": created["task_id"],
        "status": public_status(created.get("status")),
        "target_id": target.deployment_id,
        "dataset": dataset["name"],
        "backend": backend,
        "next": "Poll simulation_get(simulation_id, wait_seconds=25) until the status settles.",
    }


@tool(
    "simulation_get",
    "[R] Show a simulation's status and progress, and with include its summary metrics, non-2xx response codes with "
    "likely causes, and report file names. wait_seconds long-polls until it settles. To list simulations use "
    "resource_list(kinds=['simulation']).\n"
    "Side effects: none; resources: none; duration: up to wait_seconds; reversible: n/a.",
    GetSimulationInput,
    read_only=True,
    profile=PROFILE,
)
async def get_simulation(ctx: ToolContext, args: GetSimulationInput) -> dict[str, Any]:
    task_id = check_id(args.simulation_id, "simulation_id")
    task = await wait_for_task(ctx, task_id, args.wait_seconds)
    output: dict[str, Any] = {"simulation_id": task_id, **task_summary(task)}
    result = task.get("result")
    if "results" in args.include and result:
        output["results"] = shape_results(result)
    if "artifacts" in args.include:
        output["artifacts"] = shape_artifacts(result)
    if "response_code_issues" in args.include and task.get("status") in {"completed", "failed"}:
        diagnostics = await response_code_diagnostics(ctx, task_id, result)
        untrusted = diagnostics.pop("untrusted_text", None)
        output.update(diagnostics)
        if untrusted:
            output["untrusted_text"] = {**output.get("untrusted_text", {}), "response_errors": untrusted}
    output["next"] = (
        "Poll again with wait_seconds=25."
        if task.get("status") in ACTIVE
        else "Delete the record with resource_delete when finished."
    )
    return output


@tool(
    "simulation_cancel",
    "[D] Abort a running simulation you started; results gathered so far are kept. Already finished simulations are "
    "left unchanged. To remove the record use resource_delete.\n"
    "Side effects: stops the replay task and its traffic; resources: frees the target; duration: asynchronous, "
    "seconds; reversible: no (start a new simulation).",
    CancelSimulationInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def cancel_simulation(ctx: ToolContext, args: CancelSimulationInput) -> dict[str, Any]:
    task_id = check_id(args.simulation_id, "simulation_id")
    task = await fetch_task(ctx, task_id)
    if task.get("status") not in ACTIVE:
        return {"simulation_id": task_id, "status": public_status(task.get("status")), "note": "Already settled."}
    await ctx.api.request("POST", f"/simulation/tasks/{task_id}/stop")
    return {
        "simulation_id": task_id,
        "status": "cancelling",
        "next": "Call simulation_get(wait_seconds=10) to confirm it settled.",
    }


register_kind("simulation", list=list_simulations, get=get_simulation_entry, delete=delete_simulation)
