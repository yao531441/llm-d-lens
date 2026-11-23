"""Measured Evaluate results promoted to immutable Agentic evidence.

Run with ``python -m llm_d_lens.evaluation.benchmark_evidence`` to preview
historical ingestion; ``--apply`` writes only records that pass the same checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
import zstandard

from llm_d_lens.agentic.benchmark_context import (
    capture_benchmark_context as capture_serving_context,
)
from llm_d_lens.agentic.benchmark_context import (
    configuration_snapshot,
    enrich_model_metadata,
    persist_benchmark_records,
    validate_benchmark_context,
)
from llm_d_lens.agentic.benchmark_context import (
    positive as _positive,
)
from llm_d_lens.agentic.benchmark_context import (
    verify_benchmark_context as verify_benchmark_context,
)
from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkMetrics,
    BenchmarkRecord,
    HardwareDescriptor,
    WorkloadProfile,
)
from llm_d_lens.db.dao.evaluate_run import EvaluateRunDao

logger = logging.getLogger(__name__)

_LIFECYCLE_ARTIFACT_LIMIT = 64 * 1024 * 1024


class _DeadlineReader:
    """Decompressed archive stream that enforces the remaining read budget."""

    def __init__(self, stream: Any, deadline: float) -> None:
        self._stream = stream
        self._deadline = deadline

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self._deadline:
            raise ValueError("Lifecycle archive read timed out")
        return self._stream.read(size)


def declare_historical_hardware(run: dict, accelerator: str, vram_per_gpu_gib: float) -> None:
    """Attach an explicit operator declaration to a terminal run lacking a serving snapshot.

    Card count comes only from its checksummed saved TP/replica configuration.
    This declaration is archival context, not test-time discovery or predictive proof.
    """
    if run.get("status") not in ("succeeded", "failed") or run.get("benchmark_context"):
        raise ValueError("Hardware declarations require a terminal run without a serving snapshot")
    configuration, candidate = configuration_snapshot(run)
    if not accelerator.strip() or len(accelerator) > 64 or accelerator.strip().lower() in ("unknown", "cluster"):
        raise ValueError("An explicit accelerator model of at most 64 characters is required")
    count = candidate.decode_tp * candidate.decode_replicas
    if candidate.prefill_tp is not None and candidate.prefill_replicas is not None:
        count += candidate.prefill_tp * candidate.prefill_replicas
    hardware = HardwareDescriptor(
        accelerator=accelerator.strip(),
        vram_per_gpu_gib=_positive(vram_per_gpu_gib, "declared per-device memory"),
        gpu_count=count,
    )
    if run.get("benchmark_hardware_metadata") is not None:
        if _declared_hardware(run) != hardware:
            raise ValueError("Existing historical hardware declaration differs; it must not be silently replaced")
        return
    run["benchmark_hardware_metadata"] = {
        "schema_version": 1,
        "source": "operator-declaration",
        "declared_at": datetime.now(UTC).isoformat(),
        "configuration_checksum": configuration.checksum,
        "gpu_count_source": "saved-deployment-configuration",
        "hardware": hardware.model_dump(),
    }


def _declared_hardware(run: dict) -> HardwareDescriptor:
    """Validate saved operator attribution before using it as archival hardware context."""
    metadata = run["benchmark_hardware_metadata"]
    configuration, candidate = configuration_snapshot(run)
    if (
        run.get("benchmark_context")
        or metadata.get("schema_version") != 1
        or metadata.get("source") != "operator-declaration"
        or metadata.get("gpu_count_source") != "saved-deployment-configuration"
        or metadata.get("configuration_checksum") != configuration.checksum
    ):
        raise ValueError("Hardware declaration does not identify this saved configuration and operator source")
    if datetime.fromisoformat(metadata["declared_at"]).tzinfo is None:
        raise ValueError("Hardware declaration timestamp must include a timezone")
    hardware = HardwareDescriptor.model_validate(metadata.get("hardware"), strict=True)
    if not hardware.accelerator or hardware.accelerator.strip().lower() in ("unknown", "cluster"):
        raise ValueError("Hardware declaration must identify the accelerator model")
    _positive(hardware.vram_per_gpu_gib, "declared per-device memory")
    count = candidate.decode_tp * candidate.decode_replicas
    if candidate.prefill_tp is not None and candidate.prefill_replicas is not None:
        count += candidate.prefill_tp * candidate.prefill_replicas
    if hardware.gpu_count != count:
        raise ValueError("Declared card count disagrees with the saved deployment configuration")
    return hardware


LIFECYCLE_ARTIFACT_NAME = re.compile(r"config\.yaml|(?:summary|stage_\d+|per_request)_lifecycle_metrics\.json")


def read_lifecycle_artifact(summary_path: str | Path, name: str) -> bytes | None:
    """Read one bounded sibling artifact, loose or in its own workspace archive.

    No archive member is extracted to disk. Only lifecycle/config members are
    accepted, and ambiguous duplicate members are rejected.
    """
    if not LIFECYCLE_ARTIFACT_NAME.fullmatch(name):
        raise ValueError("Unsupported lifecycle artifact name")
    root = Path(summary_path).parent
    path = root / name
    if path.is_file():
        if path.stat().st_size > _LIFECYCLE_ARTIFACT_LIMIT:
            raise ValueError("Lifecycle artifact exceeds the 64 MiB limit")
        return path.read_bytes()
    archive = root / "workspace.tar.zst"
    if not archive.is_file():
        return None
    # Captured workspaces name members "./<name>"; stream mode never extracts a
    # member to disk and does not follow stored links.
    deadline = time.monotonic() + 15
    member_name = f"./{name}"
    payload: bytes | None = None
    try:
        with (
            archive.open("rb") as raw,
            zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as decompressed,
            tarfile.open(fileobj=_DeadlineReader(decompressed, deadline), mode="r|") as tar,
        ):
            for member in tar:
                if member.name != member_name:
                    continue
                if payload is not None:
                    raise ValueError(f"Lifecycle archive contains duplicate {member.name} members")
                if not member.isreg():
                    raise ValueError(f"Lifecycle archive member {member.name} is not a regular file")
                if member.size > _LIFECYCLE_ARTIFACT_LIMIT:
                    raise ValueError("Lifecycle artifact exceeds the 64 MiB limit")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unable to read {name} from its lifecycle archive")
                payload = extracted.read()
    except (zstandard.ZstdError, tarfile.TarError, EOFError) as error:
        raise ValueError(f"Unable to read {name} from its lifecycle archive") from error
    if payload is None:
        raise ValueError(f"Unable to read {name} from its lifecycle archive")
    return payload


def summarize_lifecycle(payload: dict, summary_path: str | Path) -> dict:
    """Normalize one native lifecycle report, retaining client output-token semantics."""
    successes = payload.get("successes") or {}
    failures = payload.get("failures") or {}
    load = payload.get("load_summary") or {}
    latency = successes.get("latency") or {}
    throughput = successes.get("throughput") or {}
    quantiles = {
        "mean": "mean",
        "min": "min",
        "max": "max",
        "p50": "median",
        "p75": "p75",
        "p90": "p90",
        "p95": "p95",
        "p99": "p99",
    }

    def distribution(values: dict, *, milliseconds: bool = False) -> dict:
        return {
            key + ("_ms" if milliseconds else ""): round(float(value) * 1000, 3) if milliseconds else value
            for key, source in quantiles.items()
            if isinstance(value := values.get(source), (int, float)) and not isinstance(value, bool)
        }

    token_lengths = {
        "input": distribution(successes.get("prompt_tokens") or successes.get("prompt_len") or {}),
        # output_len measures generated client tokens and is the throughput
        # denominator in native inference-perf; server usage may disagree.
        "output": distribution(successes.get("output_len") or successes.get("output_tokens") or {}),
    }
    success_count, failure_count = successes.get("count"), failures.get("count")
    total = success_count + failure_count if type(success_count) is int and type(failure_count) is int else None
    distributions = {
        name: distribution(latency.get(source) or {}, milliseconds=True)
        for name, source in {
            "request_latency": "request_latency",
            "ttft": "time_to_first_token",
            "itl": "inter_token_latency",
            "tpot": "time_per_output_token",
            "ntpot": "normalized_time_per_output_token",
        }.items()
    }
    return {
        "throughput_tps": throughput.get("output_tokens_per_sec"),
        "input_throughput_tps": throughput.get("input_tokens_per_sec"),
        "total_throughput_tps": throughput.get("total_tokens_per_sec"),
        "throughput_rps": throughput.get("requests_per_sec"),
        **{
            name + "_ms": distributions[source].get("p50_ms")
            for name, source in {
                "ttft": "ttft",
                "tpot": "tpot",
                "itl": "itl",
                "ntpot": "ntpot",
                "request_latency": "request_latency",
            }.items()
        },
        "latency_distributions": distributions,
        "token_distributions": token_lengths,
        "total_input_tokens": (successes.get("prompt_tokens") or {}).get("total"),
        "total_output_tokens": (successes.get("output_tokens") or {}).get("total"),
        "success_count": success_count,
        "failure_count": failure_count,
        "request_count": total,
        "success_rate": round(success_count / total * 100, 3) if total else None,
        "error_rate": round(failure_count / total * 100, 3) if total else None,
        "scheduled_request_count": load.get("count"),
        "schedule_delay": distribution(load.get("schedule_delay") or {}, milliseconds=True),
        "benchmark_time_seconds": payload.get("benchmark_time_seconds"),
        "summary_path": str(summary_path),
    }


def _native_summary(metrics: dict, stage_index: int | None) -> tuple[dict, dict | None]:
    path = metrics.get("summary_path")
    if not path:
        return metrics, None
    name = (
        f"stage_{stage_index}_lifecycle_metrics.json" if stage_index is not None else "summary_lifecycle_metrics.json"
    )
    payload = read_lifecycle_artifact(path, name)
    if payload is None:
        return metrics, None
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("Invalid native lifecycle summary")
    measured = summarize_lifecycle(raw, Path(path).with_name(name))
    for name in ("success_count", "failure_count", "request_count"):
        if metrics.get(name) is not None and measured.get(name) != metrics[name]:
            raise ValueError("Native lifecycle counts disagree with the saved sample")
    return measured, raw


async def capture_benchmark_context(run: dict) -> None:
    if run.get("harness") != "inference-perf" or run.get("parallelism") != 1:
        run["benchmark_context_error"] = "Automatic evidence attribution requires single-worker inference-perf"
        logger.warning("Evaluate %s: %s", run["id"], run["benchmark_context_error"])
        return
    await capture_serving_context(run)


def _metrics(metrics: dict) -> BenchmarkMetrics:
    latencies = metrics.get("latency_distributions") or {}
    values = {
        "ttft_p95_ms": (latencies.get("ttft") or {}).get("p95_ms"),
        "tpot_p95_ms": (latencies.get("tpot") or {}).get("p95_ms"),
        "e2e_p95_ms": (latencies.get("request_latency") or {}).get("p95_ms"),
        "throughput_tokens_per_s": metrics.get("throughput_tps"),
    }
    for name, value in values.items():
        if value is not None:
            _positive(value, name)
    if not any(value is not None for value in values.values()):
        raise ValueError("No measured P95 latency or token throughput is available")
    successes, failures = metrics.get("success_count"), metrics.get("failure_count")
    if type(successes) is not int or type(failures) is not int or min(successes, failures) < 0:
        raise ValueError("Explicit measured success and failure counts are required")
    total = successes + failures
    if total <= 0 or type(metrics.get("request_count")) is not int or metrics["request_count"] != total:
        raise ValueError("Measured request counts are missing or inconsistent")
    return BenchmarkMetrics(**values, error_rate=failures / total)


def _concurrency(metrics: dict, stage_index: int | None, native: dict | None = None) -> float:
    if native is not None:
        # Sum of request lifetimes / measured stage duration is mean in-flight
        # concurrency. Stage aggregates avoid guessing stage IDs on unlabeled rows.
        duration = _positive(metrics.get("benchmark_time_seconds"), "benchmark_time_seconds")
        lifetime = 0.0
        for outcome, count_name in (("successes", "success_count"), ("failures", "failure_count")):
            count = metrics.get(count_name)
            if type(count) is not int or count < 0:
                raise ValueError("Measured request counts are missing")
            if count:
                section = native.get(outcome) or {}
                latency = (
                    (section.get("latency") or {}).get("request_latency")
                    if outcome == "successes"
                    else section.get("request_latency")
                )
                mean = _positive((latency or {}).get("mean"), "measured mean request latency")
                if mean > duration * 1.001 + 0.001:
                    raise ValueError("Mean request latency exceeds the measured stage duration")
                lifetime += count * mean
        return _positive(lifetime / duration, "measured mean concurrency")
    path = metrics.get("summary_path")
    if not path:
        raise ValueError("Measured concurrency requires per-request lifecycle artifacts")
    payload = read_lifecycle_artifact(path, "per_request_lifecycle_metrics.json")
    if payload is None:
        raise ValueError("Per-request lifecycle artifacts are missing")
    rows = json.loads(payload)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid per-request lifecycle artifacts")
    if stage_index is not None and any(type(row.get("stage_id")) is not int for row in rows):
        raise ValueError("Per-request stage identities are missing")
    rows = (
        [row for row in rows if row.get("stage_id") == stage_index]
        if stage_index is not None
        else [row for row in rows if row.get("stage_id") != -1]
    )
    if len(rows) != metrics.get("request_count") or not rows:
        raise ValueError("Per-request coverage is incomplete")
    duration = _positive(metrics.get("benchmark_time_seconds"), "benchmark_time_seconds")
    total = 0.0
    successes = 0
    identities = set()
    for row in rows:
        if "error" not in row:
            raise ValueError("Per-request completion status is missing")
        successes += row["error"] is None
        identity = row.get("request_id")
        if identity is not None:
            if not isinstance(identity, str) or identity in identities:
                raise ValueError("Per-request identities are invalid or duplicated")
            identities.add(identity)
        start, end = row.get("start_time"), row.get("end_time")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (start, end)):
            raise ValueError("Per-request timestamps are invalid")
        if end < start:
            raise ValueError("Per-request completion precedes its start")
        total += end - start
    if successes != metrics.get("success_count") or len(rows) - successes != metrics.get("failure_count"):
        raise ValueError("Per-request outcomes disagree with the lifecycle summary")
    if max(row["end_time"] for row in rows) - min(row["start_time"] for row in rows) > duration * 1.001 + 0.001:
        raise ValueError("Per-request timestamps exceed the measured duration")
    return _positive(total / duration, "measured mean concurrency")


def _workload(
    run: dict, metrics: dict, concurrency: Any, stage_index: int | None, native: dict | None = None
) -> WorkloadProfile:
    tokens = metrics.get("token_distributions") or {}
    mean_input = _positive((tokens.get("input") or {}).get("mean"), "measured mean input tokens")
    mean_output = _positive((tokens.get("output") or {}).get("mean"), "measured mean output tokens")
    if run.get("matrix"):
        shared_prefix = 0.0  # Generated matrix inputs are random, not shared-prefix traffic.
    elif run.get("shared_prefix"):
        spec = run["shared_prefix"]
        if spec.get("enable_multi_turn_chat"):
            raise ValueError("Multi-turn prefix reuse is not described by a single prefix ratio")
        shared_prefix = _positive(spec.get("system_prompt_len"), "system_prompt_len") / (
            spec["system_prompt_len"] + _positive(spec.get("question_len"), "question_len")
        )
    else:
        workload = yaml.safe_load(run["workload_yaml"]) if run.get("workload_yaml") else None
        if not workload and metrics.get("summary_path"):
            payload = read_lifecycle_artifact(metrics["summary_path"], "config.yaml")
            workload = yaml.safe_load(payload) if payload is not None else None
        data = (workload or {}).get("data") or {}
        if data.get("type") != "random":
            raise ValueError("The saved workload does not establish its shared-prefix semantics")
        shared_prefix = 0.0
        stages = (workload.get("load") or {}).get("stages") or []
        if len(stages) > 1:
            raise ValueError("Mixed workload stages must not be promoted as one benchmark sample")
        if stages:
            concurrency = stages[0].get("concurrency_level")
    return WorkloadProfile(
        mean_input_tokens=mean_input,
        p95_input_tokens=_positive((tokens.get("input") or {}).get("p95"), "measured P95 input tokens"),
        mean_output_tokens=mean_output,
        concurrency=_positive(concurrency, "configured concurrency")
        if concurrency is not None
        else _concurrency(metrics, stage_index, native),
        request_rate=_positive(metrics.get("throughput_rps"), "measured request rate"),
        shared_prefix_ratio=shared_prefix,
        prefill_fraction=mean_input / (mean_input + mean_output),
    )


def normalize_evaluate_run(run: dict) -> tuple[list[BenchmarkRecord], list[dict]]:
    """Normalize saved evidence only; never consult current planning or live cluster state."""
    issues: list[dict] = []
    try:
        if run.get("kind") != "benchmark" or run.get("status") not in ("succeeded", "failed"):
            raise ValueError("Only succeeded or failed Evaluate benchmark runs are eligible")
        if run.get("harness") != "inference-perf" or type(run.get("parallelism")) is not int or run["parallelism"] != 1:
            raise ValueError("Evidence requires inference-perf with one worker and attributable lifecycle reports")
        run_id = str(UUID(run["id"]))
        timestamp, model, hardware, runtime, candidate = validate_benchmark_context(run, allow_saved_configuration=True)
        if run.get("benchmark_hardware_metadata") is not None:
            hardware = _declared_hardware(run)
        if run["status"] == "failed" and not run.get("error"):
            raise ValueError("Failed benchmark evidence requires an explicit failure reason")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return [], [{"sample": "run", "reason": str(error)}]

    samples = []
    if run.get("matrix"):
        for point_index, point in enumerate(run.get("matrix_results") or []):
            for stage_index, stage in enumerate(point.get("stage_metrics") or []):
                samples.append((f":matrix:{point_index}:stage:{stage_index}", stage, stage_index, point.get("status")))
    elif run.get("shared_prefix"):
        samples = [
            (f":stage:{index}", stage, index, run["status"])
            for index, stage in enumerate(run.get("rate_stage_results") or [])
        ]
    else:
        samples = [("", {"metrics": run.get("metrics")}, None, run["status"])]
    if not samples:
        return [], [{"sample": "run", "reason": "No individually measured workload stages are available"}]
    records = []
    for suffix, sample, stage_index, status in samples:
        try:
            if status not in ("succeeded", "failed"):
                raise ValueError("Workload sample has no terminal outcome")
            metrics = sample.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("Measured lifecycle summary is missing")
            metrics, native = _native_summary(metrics, stage_index)
            benchmark_id = f"evaluate:{run_id}{suffix}"
            workload = _workload(run, metrics, sample.get("concurrency"), stage_index, native)
            record = BenchmarkRecord(
                benchmark_id=benchmark_id,
                timestamp=timestamp,
                outcome=status,
                model=model,
                hardware=hardware,
                runtime=runtime,
                workload=workload,
                configuration=candidate,
                metrics=_metrics(metrics),
                failure_reason=run.get("error") if status == "failed" else None,
                provenance={
                    "schema_version": 1,
                    "attribution": "serving-snapshot" if run.get("benchmark_context") else "saved-configuration",
                    "configuration_checksum": run["deployment_configuration"]["checksum"],
                    "benchmark_started_at": run["benchmark_started_at"],
                    "finished_at": run["finished_at"],
                    "serving_snapshot": run.get("benchmark_context"),
                    "model_enrichment": run.get("benchmark_model_metadata"),
                    "hardware_enrichment": run.get("benchmark_hardware_metadata"),
                    "collection_error": run.get("benchmark_context_error"),
                    "summary_path": metrics.get("summary_path"),
                    "measurement_source": "native-lifecycle" if native is not None else "saved-metrics",
                },
            )
            records.append(record)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            issues.append({"sample": suffix or "run", "reason": str(error)})
    return records, issues


def ingest_evaluate_run(run: dict, *, apply: bool = True) -> dict:
    records, issues = normalize_evaluate_run(run)
    report = persist_benchmark_records(records, issues, source_id=run["id"], apply=apply)
    report["metadata"] = {
        "serving_snapshot": bool((run.get("benchmark_context") or {}).get("verified_at")),
        "model_enrichment": run.get("benchmark_model_metadata"),
        "hardware_enrichment": run.get("benchmark_hardware_metadata"),
        "collection_error": run.get("benchmark_context_error"),
        "enrichment_error": run.get("benchmark_model_metadata_error"),
        "missing_fields": sorted(
            {
                f"{section}.{name}"
                for record in records
                for section in ("model", "hardware", "runtime")
                for name, value in getattr(record, section).model_dump().items()
                if value is None
            }
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist eligible evidence; default is read-only preview")
    parser.add_argument("--run-id", action="append", type=UUID, help="Only process these saved Evaluate run IDs")
    parser.add_argument(
        "--enrich", action="store_true", help="Look up public model descriptors; never query live serving state"
    )
    parser.add_argument("--hardware-accelerator", help="Operator-declared historical accelerator model")
    parser.add_argument("--hardware-vram-per-gpu-gib", type=float, help="Operator-declared per-device memory in GiB")
    parser.add_argument(
        "--hardware-count-from-configuration",
        action="store_true",
        help="Acknowledge using saved TP times replicas as configured, not measured, card count",
    )
    args = parser.parse_args()
    hardware_options = (
        args.hardware_accelerator is not None,
        args.hardware_vram_per_gpu_gib is not None,
        args.hardware_count_from_configuration,
    )
    if any(hardware_options) and (not all(hardware_options) or not args.run_id):
        parser.error("Hardware declarations require explicit --run-id filters and all three --hardware-* options")
    dao = EvaluateRunDao()
    if args.run_id:
        runs = []
        for run_id in args.run_id:
            run = dao.get(str(run_id))
            if run is None:
                parser.error(f"Evaluate run not found: {run_id}")
            runs.append(run)
    else:
        runs = dao.list()
    payloads = []
    for run in runs:
        payload = run.model_dump(mode="json")
        if all(hardware_options):
            try:
                declare_historical_hardware(payload, args.hardware_accelerator, args.hardware_vram_per_gpu_gib)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                parser.error(f"Invalid hardware declaration for {run.id}: {error}")
        payloads.append((run, payload))
    failed = False
    for run, payload in payloads:
        if args.enrich:
            asyncio.run(enrich_model_metadata(payload))
        report = ingest_evaluate_run(payload, apply=args.apply)
        if args.apply and run.status in ("succeeded", "failed", "cancelled"):
            payload["benchmark_evidence"] = report
            dao.save(type(run).model_validate(payload))
        print(json.dumps({"run_id": run.id, "apply": args.apply, **report}))
        failed = failed or report["status"] == "error"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
