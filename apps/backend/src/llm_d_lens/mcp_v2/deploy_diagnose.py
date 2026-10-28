"""Rule-based deployment diagnosis from execution status, pods, scheduler events and recent logs."""

from __future__ import annotations

import re
from typing import Any

from .cluster_info import cluster_summary
from .context import ToolContext
from .deploy_status import default_container, now_iso
from .deployment_refs import DeploymentRef, get_run, latest_case
from .lens_api import LensApiError

IMAGE_PULL_REASONS = {"ImagePullBackOff", "ErrImagePull", "InvalidImageName", "ErrImageNeverPull"}
LOG_TAIL = 150
MAX_LOG_PODS = 3
EXCERPT_LINES = 3
EXCERPT_CHARS = 240

# (code, title, suggested fix, regex) applied to recent pod logs of unhealthy pods.
LOG_RULES = (
    (
        "model_access_denied",
        "The model repository refused access (gated or private model).",
        "Make sure the cluster Hugging Face secret may read this model; see list_models / get_cluster.",
        re.compile(r"gated repo|401 Client Error|403 Client Error|Access to model .* is restricted", re.I),
    ),
    (
        "model_not_found",
        "The model could not be found in the repository.",
        "Check the model id spelling in the spec, then plan_deployment again.",
        re.compile(r"Repository Not Found|RepositoryNotFoundError|404 Client Error|is not a local folder", re.I),
    ),
    (
        "disk_full",
        "A node ran out of disk space while downloading or writing the model.",
        "Use a model-cache volume with enough space (list_storage) or a smaller model.",
        re.compile(r"No space left on device", re.I),
    ),
    (
        "model_download_failed",
        "The model download failed on the network.",
        "Check cluster egress/proxy, pre-load the model into a cache, then create_deployment(from_deployment_id=...).",
        re.compile(r"Max retries exceeded|Connection (error|reset|refused|timed out)|Temporary failure in name", re.I),
    ),
    (
        "accelerator_oom",
        "The model server ran out of accelerator memory.",
        "Lower spec.engine.max_model_len or raise tensor_parallel_size, then create_deployment again.",
        re.compile(r"out of memory|OutOfMemory|OUT_OF_DEVICE_MEMORY|OOM", re.I),
    ),
)
EVENT_UNSCHEDULABLE = re.compile(r"FailedScheduling|Insufficient|didn't match Pod's node affinity|Unschedulable", re.I)
EVENT_READINESS = re.compile(r"Readiness probe failed", re.I)


def _finding(code: str, severity: str, title: str, evidence: list[str], fix: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "title": title, "evidence": evidence, "suggested_fix": fix}


async def _events_text(ctx: ToolContext, ref: DeploymentRef) -> str:
    if not ref.run_id:
        return ""
    case = latest_case(await get_run(ctx, ref.run_id) or {})
    if not case or not case.get("id"):
        return ""
    try:
        page = await ctx.api.request(
            "GET", f"/deployments/runs/{ref.run_id}/cases/{case['id']}/logs", params={"limit": 200}, timeout=30
        )
    except LensApiError:
        return ""
    return "\n".join(
        str(entry.get("message") or "") for entry in page.get("entries") or [] if entry.get("source") == "events"
    )


async def _pod_logs(ctx: ToolContext, execution_id: str, pod: dict[str, Any]) -> str:
    texts = []
    for previous in (False, True) if pod["restarts"] else (False,):
        try:
            data = await ctx.api.request(
                "GET",
                f"/deployments/executions/{execution_id}/pods/{pod['name']}/logs",
                params={"container": default_container(pod), "tail": LOG_TAIL, "previous": previous},
                timeout=40,
            )
        except LensApiError:
            continue
        if data.get("success"):
            texts.append(data.get("logs") or "")
    return "\n".join(texts)


def _excerpt(text: str, pattern: re.Pattern[str]) -> list[str]:
    lines = [line.strip()[:EXCERPT_CHARS] for line in text.splitlines() if pattern.search(line)]
    return lines[-EXCERPT_LINES:]


def _pod_findings(pod: dict[str, Any], status: str, events: str) -> list[dict[str, Any]]:
    name, reason, found = pod["name"], pod["reason"], []
    facts = [f"pod {name} ({pod['role']}): phase={pod['phase']}, reason={reason}, restarts={pod['restarts']}"]
    if reason in IMAGE_PULL_REASONS:
        found.append(
            _finding(
                "image_pull_error",
                "critical",
                "The container image cannot be pulled.",
                facts,
                "Check spec.runtime_image and registry access; fix it and create_deployment(from_deployment_id=...).",
            )
        )
    elif reason == "OOMKilled":
        found.append(
            _finding(
                "oom_killed",
                "critical",
                "A container was killed for exceeding its memory limit.",
                facts,
                "Lower max_model_len or raise tensor_parallel_size, then create_deployment(from_deployment_id=...).",
            )
        )
    elif reason == "CrashLoopBackOff" or (pod["restarts"] >= 3 and not pod["ready"]):
        found.append(
            _finding(
                "crash_loop",
                "critical",
                "A container keeps crashing and restarting.",
                facts,
                f"Call get_logs(pod='{name}', previous=true) to see why the last run exited.",
            )
        )
    elif pod["phase"] == "Pending":
        unschedulable = bool(EVENT_UNSCHEDULABLE.search(events))
        found.append(
            _finding(
                "unschedulable" if unschedulable else "pod_pending",
                "critical" if unschedulable else "warning",
                "No node can run this pod right now." if unschedulable else "The pod has not been scheduled yet.",
                facts + (["scheduler events report the pod cannot be placed"] if unschedulable else []),
                "Free accelerators (stop_deployment, delete_resources) or reduce replicas / tensor_parallel_size.",
            )
        )
    elif pod["phase"] == "Running" and not pod["ready"]:
        probing = bool(EVENT_READINESS.search(events))
        found.append(
            _finding(
                "readiness_probe_failing" if probing else "not_ready",
                "warning" if probing or status != "starting" else "info",
                "The readiness probe is failing." if probing else "The pod is running but not ready yet.",
                facts,
                "Model loading can take minutes; if it persists call get_logs for this pod.",
            )
        )
    return found


async def diagnose(ctx: ToolContext, ref: DeploymentRef, view: dict[str, Any]) -> dict[str, Any]:
    status, pods, execution = view["status"], view["pods"], view["execution"] or {}
    events = await _events_text(ctx, ref)
    findings: list[dict[str, Any]] = []
    untrusted: list[dict[str, Any]] = []

    if status == "failed":
        findings.append(
            _finding(
                "deployment_failed",
                "critical",
                "Lens reports that the deployment failed.",
                ["execution status is failed"],
                "Fix the cause below, then create_deployment with from_deployment_id and a spec_patch.",
            )
        )
    if view["failure"]:
        untrusted.append({"source": "lens_failure", "lines": [view["failure"]]})

    for pod in pods:
        findings.extend(_pod_findings(pod, status, events))
    if events and EVENT_UNSCHEDULABLE.search(events):
        untrusted.append({"source": "scheduler_events", "lines": _excerpt(events, EVENT_UNSCHEDULABLE)})

    unhealthy = [p for p in pods if not p["ready"] and p["phase"] != "Pending"][:MAX_LOG_PODS]
    for pod in unhealthy:
        text = await _pod_logs(ctx, ref.execution_id, pod)
        for code, title, fix, pattern in LOG_RULES:
            lines = _excerpt(text, pattern)
            if lines and not any(f["code"] == code for f in findings):
                findings.append(_finding(code, "critical", title, [f"pod {pod['name']} logs match this failure"], fix))
                untrusted.append({"source": f"pod/{pod['name']}", "lines": lines})

    needed = _accelerators_needed(ref, execution)
    cluster_id = execution.get("cluster_id") or (ref.row.cluster_id if ref.row else None)
    if needed and cluster_id and (status in ("pending", "failed") or any(p["phase"] == "Pending" for p in pods)):
        free = (await cluster_summary(ctx, cluster_id)).get("accelerators_free")
        if free is not None and needed > free:
            findings.append(
                _finding(
                    "insufficient_capacity",
                    "critical",
                    "The cluster has fewer free accelerators than this deployment needs.",
                    [f"needs {needed} accelerators, {free} free"],
                    "Free accelerators (list_resources, stop_deployment) or plan a smaller topology.",
                )
            )

    actionable = [f for f in findings if f["severity"] != "info"]
    return {
        "deployment_id": ref.public_id,
        "status": status,
        "healthy": status == "ready" and not actionable,
        "findings": findings,
        "observed_at": now_iso(),
        "untrusted_text": untrusted,
        "next": _next(status, findings),
    }


def _accelerators_needed(ref: DeploymentRef, execution: dict[str, Any]) -> int | None:
    topology = ((ref.row.meta.get("spec") if ref.row else None) or {}).get("topology") or {}
    if topology:
        return sum(
            (role.get("replicas") or 1) * (role.get("tensor_parallel_size") or 1) for role in topology.values() if role
        )
    if execution.get("replicas") and execution.get("tensor_parallel_size"):
        return execution["replicas"] * execution["tensor_parallel_size"]
    return None


def _next(status: str, findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No problem found." if status == "ready" else "Nothing wrong yet; call get_deployment(wait_seconds=25)."
    if any(f["code"] == "insufficient_capacity" for f in findings):
        return "Call get_cluster to see free accelerators, then stop or delete something, or plan a smaller topology."
    return "Apply the suggested_fix of the first critical finding; get_logs shows the raw container output."
