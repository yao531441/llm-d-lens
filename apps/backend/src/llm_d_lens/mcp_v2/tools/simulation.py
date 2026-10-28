"""Simulation profile: replay a trace dataset against one of the caller's own deployments."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..errors import ToolError
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..registry import tool
from ..simulation_target import resolve_target
from ..simulation_view import (
    ACTIVE,
    delete_simulation,
    describe_simulation,
    estimate_replay,
    fetch_task,
    list_other_simulations,
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
        "so no trial run is needed before start_simulation.",
    )
    dataset: str | None = Field(None, description="Dataset name from this list; required with include=['estimate'].")
    duration_seconds: int = Field(60, ge=5, le=7200, description="Replay length used for the estimate.")
    scale_factor: float = Field(1.0, gt=0, le=50, description="Replay speed-up used for the estimate.")


class PrepareTraceDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="Dataset name from list_trace_datasets.")
    force: bool = Field(False, description="Download again even if the file is already present.")


class StartSimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(
        description="A deployment you created (dep-...). Targets owned by someone else are refused, and model "
        "services (svc-...) are not supported."
    )
    dataset: str = Field(description="Downloaded dataset name from list_trace_datasets; it fixes the scenario.")
    duration_seconds: int = Field(60, ge=5, le=7200, description="How long the replay runs.")
    scale_factor: float = Field(
        1.0, gt=0, le=50, description="Speed-up of the trace; only datasets whose format supports it accept != 1."
    )
    backend: str | None = Field(None, description="Replay backend; default is the dataset's preferred one.")
    name: str | None = Field(None, max_length=120, description="Display name; generated when omitted.")
    ttl_hours: int = Field(8, ge=1, le=72, description="How long the record is kept before it is eligible for cleanup.")
    client_request_id: str | None = Field(None, description="Retry key: repeating it returns the first result.")


class GetSimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(description="sim-... id from start_simulation, or a Lens task id from list_resources.")
    include: list[Literal["results", "response_code_issues", "artifacts"]] = Field(
        default_factory=lambda: ["results"],
        description="results = summary metrics; response_code_issues = non-2xx codes with likely causes and samples; "
        "artifacts = report file names.",
    )
    wait_seconds: int = Field(0, ge=0, le=25, description="Wait up to this long for the simulation to settle.")


class CancelSimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(description="sim-... id of a simulation you started.")
    reason: str | None = Field(None, max_length=300, description="Recorded in the audit log.")


def _scale_formats(backend: dict[str, Any]) -> list[str]:
    formats = (backend.get("capabilities") or {}).get("trace_format_capabilities") or {}
    return [name for name, caps in formats.items() if (caps or {}).get("supports_scale_factor")]


@tool(
    "list_trace_datasets",
    "[R] List the trace datasets, scenarios and replay backends a simulation can use, with whether each dataset is "
    "downloaded. include=['estimate'] with a dataset gives the request count and peak rate of a replay, replacing a "
    "dry run before start_simulation. To fetch a missing dataset use prepare_trace_dataset.\n"
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
            result["next"] = f"Run prepare_trace_dataset for '{args.dataset}' before estimating."
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
        raise ToolError("not_found", f"No dataset '{name}'.", field="dataset", hint="Call list_trace_datasets.")
    return entry


@tool(
    "prepare_trace_dataset",
    "[W] Download and validate a registered trace dataset so it can be replayed. Returns at once when the dataset is "
    "already present. Find names with list_trace_datasets; start the replay with start_simulation.\n"
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
        "next": "Call start_simulation." if status == "ready" else "Retry prepare_trace_dataset; check the Lens logs.",
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
    for row in ctx.ledger.list_resources(owner=ctx.owner, kind="simulation"):
        if row.meta.get("target_execution_id") != execution_id or not row.lens_execution_id:
            continue
        try:
            task = await fetch_task(ctx, row.lens_execution_id)
        except ToolError:
            continue
        if task.get("status") in ACTIVE:
            raise ToolError(
                "conflict_state",
                f"{row.id} is already replaying against this deployment.",
                field="target_id",
                hint="Wait for it with get_simulation, or stop it with cancel_simulation.",
            )


@tool(
    "start_simulation",
    "[S] Replay a trace dataset against a deployment you created and return a sim-... id; poll get_simulation. It "
    "loads the target for the whole duration, so only your own deployments are accepted and only one replay may run "
    "per deployment. For comparing configurations under controlled load use the experiment tools instead.\n"
    "Side effects: sends request traffic to the target; resources: target accelerators and a Lens replay task; "
    "duration: asynchronous, as long as duration_seconds; reversible: stop it with cancel_simulation.",
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
            hint="Call prepare_trace_dataset first.",
        )
    backend, supports_range = await _pick_backend(ctx, dataset, args.backend, args.scale_factor)
    await _ensure_target_idle(ctx, target.execution_id)
    name = args.name or f"{dataset['name']}-{target.row.name or target.row.id}"[:120]
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
    row = ctx.ledger.add_resource(
        kind="simulation",
        owner=ctx.owner,
        cluster_id=target.cluster_id,
        name=name,
        state="creating",
        meta={
            "target_id": target.row.id,
            "target_execution_id": target.execution_id,
            "dataset": dataset["name"],
            "expires_at": time.time() + args.ttl_hours * 3600,
        },
    )
    try:
        created = await ctx.api.request("POST", "/simulation/tasks", body=body, timeout=120)
        task_id = created["task_id"]
    except LensApiError as error:
        if 400 <= error.status < 500:
            ctx.ledger.set_state(row.id, "deleted")
            raise
        raise _outcome_unknown(row.id) from error
    except Exception as error:
        raise _outcome_unknown(row.id) from error
    ctx.ledger.bind_execution(row.id, task_id)
    ctx.ledger.set_state(row.id, "active")
    return {
        "simulation_id": row.id,
        "status": public_status(created.get("status")),
        "target_id": target.row.id,
        "dataset": dataset["name"],
        "backend": backend,
        "next": "Poll get_simulation(simulation_id, wait_seconds=25) until the status settles.",
    }


def _outcome_unknown(simulation_id: str) -> ToolError:
    """The replay may or may not exist in Lens, so the ledger row is kept in ``creating`` instead of deleted."""
    return ToolError(
        "upstream_unavailable",
        "Lens did not confirm whether the simulation was created.",
        hint=f"Call get_simulation('{simulation_id}') in a minute: it links the record to the replay if it started. "
        "Do not call start_simulation again until then.",
        simulation_id=simulation_id,
    )


async def _adopt_unbound(ctx: ToolContext, simulation_id: str) -> None:
    """Link a ``creating`` record to its Lens task when the create response was lost (single exact name match only)."""
    row = ctx.ledger.get(simulation_id) if simulation_id.startswith("sim-") else None
    if row is None or row.kind != "simulation" or row.owner != ctx.owner or row.lens_execution_id:
        return
    page = await ctx.api.request("GET", "/simulation/tasks", params={"limit": 100})
    candidates = [
        task
        for task in page.get("tasks") or []
        if task.get("name") == row.name and ctx.ledger.find_by_execution(task["id"]) is None
    ]
    if len(candidates) == 1:
        ctx.ledger.bind_execution(row.id, candidates[0]["id"])
        ctx.ledger.set_state(row.id, "active")


def _resolve_simulation(ctx: ToolContext, simulation_id: str) -> tuple[str, bool, str | None]:
    """Return ``(lens task id, owned, public id)``; reading another user's task by its Lens id is allowed."""
    if simulation_id.startswith("sim-"):
        row = ctx.ledger.get(simulation_id)
        if row is None or row.kind != "simulation":
            raise ToolError("not_found", f"No simulation '{simulation_id}'.", field="simulation_id")
        if row.owner != ctx.owner:
            raise ToolError("not_owner", f"{simulation_id} belongs to another user.", field="simulation_id")
        if not row.lens_execution_id:
            raise ToolError(
                "conflict_state",
                f"{simulation_id} has no confirmed Lens task yet.",
                hint="Retry get_simulation shortly; if it stays unlinked the replay did not start, so delete the "
                "record with delete_resources and start again.",
            )
        return row.lens_execution_id, True, row.id
    row = ctx.ledger.find_by_execution(simulation_id)
    return simulation_id, bool(row and row.owner == ctx.owner), row.id if row else None


@tool(
    "get_simulation",
    "[R] Show a simulation's status and progress, and with include its summary metrics, non-2xx response codes with "
    "likely causes, and report file names. wait_seconds long-polls until it settles. To list simulations use "
    "list_resources(kinds=['simulation']).\n"
    "Side effects: none; resources: none; duration: up to wait_seconds; reversible: n/a.",
    GetSimulationInput,
    read_only=True,
    profile=PROFILE,
)
async def get_simulation(ctx: ToolContext, args: GetSimulationInput) -> dict[str, Any]:
    await _adopt_unbound(ctx, args.simulation_id)
    task_id, owned, public_id = _resolve_simulation(ctx, args.simulation_id)
    task = await wait_for_task(ctx, task_id, args.wait_seconds)
    output: dict[str, Any] = {"simulation_id": public_id or task_id, "owned": owned, **task_summary(task)}
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
        else "Delete the record with delete_resources when finished."
    )
    return output


@tool(
    "cancel_simulation",
    "[D] Abort a running simulation you started; results gathered so far are kept. Already finished simulations are "
    "left unchanged. To remove the record use delete_resources.\n"
    "Side effects: stops the replay task and its traffic; resources: frees the target; duration: asynchronous, "
    "seconds; reversible: no (start a new simulation).",
    CancelSimulationInput,
    read_only=False,
    destructive=True,
    idempotent=True,
    profile=PROFILE,
)
async def cancel_simulation(ctx: ToolContext, args: CancelSimulationInput) -> dict[str, Any]:
    if not args.simulation_id.startswith("sim-"):
        raise ToolError("invalid_argument", "Pass a sim-... id.", field="simulation_id")
    task_id, _, public_id = _resolve_simulation(ctx, args.simulation_id)
    task = await fetch_task(ctx, task_id)
    if task.get("status") not in ACTIVE:
        return {"simulation_id": public_id, "status": public_status(task.get("status")), "note": "Already settled."}
    await ctx.api.request("POST", f"/simulation/tasks/{task_id}/stop")
    return {
        "simulation_id": public_id,
        "status": "cancelling",
        "next": "Call get_simulation(wait_seconds=10) to confirm it settled.",
    }


register_kind(
    "simulation",
    delete=delete_simulation,
    describe=describe_simulation,
    list_others=list_other_simulations,
)
