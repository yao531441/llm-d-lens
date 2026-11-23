"""Tests for the deployment management API: list, metadata edit, and delete."""

import importlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.responses import JSONResponse
from sqlalchemy import event
from sqlalchemy.orm import Session

from llm_d_lens.auth.records import GroupRecord, GroupRoleBindingRecord, RoleRecord, UserRecord, UserRoleBindingRecord
from llm_d_lens.auth.service import default_service
from llm_d_lens.db.dao.base import ConcurrentUpdateError
from llm_d_lens.db.dao.cluster import ClusterDao
from llm_d_lens.db.dao.deployment_batch import DeploymentBatchDao
from llm_d_lens.db.dao.group import GroupDao
from llm_d_lens.db.dao.role import RoleDao
from llm_d_lens.db.dao.user import UserDao
from llm_d_lens.deploy.contracts import (
    ConfigurationArtifact,
    DeployableConfiguration,
    DeploymentArtifact,
    DeploymentCase,
    DeploymentCaseStatus,
    DeploymentCreateRequest,
    DeploymentEndpoint,
    DeploymentExecution,
    DeploymentMetadata,
    DeploymentMetadataUpdateRequest,
    DeploymentRun,
    DeploymentRunStatus,
    DeploymentScaleRequest,
    DeploymentStatus,
    VersionedPayload,
)
from llm_d_lens.deploy.run_store import JsonDeploymentRunStore

router = importlib.import_module("llm_d_lens.deploy.router")
executions = importlib.import_module("llm_d_lens.deploy.executions")
usage = importlib.import_module("llm_d_lens.deploy.usage")


class CleanFailedError(Exception):
    """Provider cleanup failure used to prove records survive a failed delete."""


class FakeDeploymentService:
    """Minimal deployment service that records cleanup calls."""

    def __init__(self, *, fail: bool = False, cleaned_up: bool = False) -> None:
        self.fail = fail
        self.cleaned_up = cleaned_up
        self.cleaned: list[str] = []

    async def clean(self, execution, *, preserve_rendered_overlay: bool = False):
        if self.fail:
            raise CleanFailedError("namespace could not be deleted")
        self.cleaned.append(execution.execution_id)
        status = DeploymentStatus.CLEANED_UP if self.cleaned_up else DeploymentStatus.CLEANED
        return execution.model_copy(update={"status": status, "endpoint": None})


def _artifact(provider_ref: str = "optimized-baseline") -> ConfigurationArtifact:
    return ConfigurationArtifact(
        schema_version="v1",
        kind="helm-values",
        provider_ref=provider_ref,
        checksum="checksum",
        content=json.dumps({"model": {"name": "Qwen/Qwen3-0.6B"}}),
    )


def _case(run_id: str, ordinal: int, execution_id: str, status: DeploymentCaseStatus) -> DeploymentCase:
    return DeploymentCase(
        id=f"case-{ordinal}",
        run_id=run_id,
        ordinal=ordinal,
        source_configuration_ordinal=ordinal,
        provider_ref="optimized-baseline",
        component="modelservice",
        create_request=DeploymentCreateRequest(
            configuration_artifacts=[_artifact()],
            deployment_policy=VersionedPayload(schema_version="v1", value={"deployment_name": f"qwen-{ordinal}"}),
        ),
        status=status,
        execution_id=execution_id,
    )


def _execution(execution_id: str, status: DeploymentStatus, namespace: str) -> DeploymentExecution:
    return DeploymentExecution(
        execution_id=execution_id,
        request_id="request-1",
        status=status,
        artifact=DeploymentArtifact(
            artifact_hash="hash",
            configuration_artifact_ids=["artifact-1"],
            manifest_ref="vllm/manifest.yaml",
            rendered_payload=VersionedPayload(schema_version="v1"),
        ),
        endpoint=(
            DeploymentEndpoint(url=f"http://vllm.{namespace}.svc:8000") if status == DeploymentStatus.READY else None
        ),
        namespace=namespace,
        configuration_artifacts=[_artifact()],
        provenance={"cluster_server_id": "cluster-1", "cluster_session_id": "session-1"},
    )


@pytest.mark.parametrize(
    "prefix_args,expected_prefix",
    [
        ([], None),
        (["--enable-prefix-caching"], True),
        (["--no-enable-prefix-caching"], False),
        (["--enable-prefix-caching=false"], False),
        (["--enable-prefix-caching=unknown"], None),
    ],
)
def test_serving_metadata_reads_rendered_source_composition_parameters(tmp_path, prefix_args, expected_prefix):
    case = _case("run-source", 1, "execution-source", DeploymentCaseStatus.READY)
    case.provider_ref = "source-composition"
    execution = _execution("execution-source", DeploymentStatus.READY, "serving")
    documents = []
    for role, replicas, parallelism in [("prefill", 1, 4), ("decode", 2, 2)]:
        documents.append(
            {
                "kind": "Deployment",
                "metadata": {"name": role},
                "spec": {
                    "replicas": replicas,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "modelserver",
                                    "image": "ghcr.io/llm-d/llm-d-xpu:v1.0.0",
                                    "command": ["vllm", "serve"],
                                    "args": [
                                        "Qwen/Qwen3-14B",
                                        f"--tensor-parallel-size={parallelism}",
                                        "--max-model-len=8192",
                                        "--max-num-seqs",
                                        "32",
                                        "--gpu-memory-utilization=0.9",
                                        *prefix_args,
                                        "--max-num-batched-tokens=8192",
                                    ],
                                }
                            ]
                        }
                    },
                },
            }
        )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("\n---\n".join(json.dumps(document) for document in documents))
    execution.artifact.manifest_ref = str(manifest)

    metadata = executions.serving_metadata(case, execution)

    assert metadata["tensor_parallel_size"] == 2
    assert metadata["max_model_len"] == 8192
    assert metadata["max_num_seqs"] == "32"
    assert metadata["gpu_memory_utilization"] == "0.9"
    assert metadata["enable_prefix_caching"] is expected_prefix
    assert metadata["max_num_batched_tokens"] == "8192"
    assert metadata["image"] == "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    assert metadata["replicas"] == 2
    assert metadata["prefill_replicas"] == 1

    case.create_request.configuration_artifacts[0].content = json.dumps({"decode": {"maxModelLen": 16384}})
    assert executions.serving_metadata(case, execution)["max_model_len"] == 16384
    manifest.write_text("invalid: [")
    assert executions.serving_metadata(case, execution)["max_model_len"] == 16384
    assert executions.serving_metadata(case, execution)["max_num_seqs"] is None


@pytest.mark.parametrize("conflicts", [1, 2, 3])
def test_execution_save_reloads_after_concurrent_metadata_update(tmp_path, conflicts: int) -> None:
    """Retry rolled-back writes with fresh metadata, but bound persistent conflicts."""
    store = JsonDeploymentRunStore(tmp_path)
    execution = _execution("execution-race", DeploymentStatus.DEPLOYING, "serving")
    store.save_execution(execution)
    updates = []
    updating = False

    def concurrent_update(session: Session, row: object) -> None:
        nonlocal updating
        if updating or getattr(row, "__tablename__", None) != "deployment_evidences" or len(updates) >= conflicts:
            return
        updating = True
        try:
            name = f"Edited concurrently {len(updates) + 1}"
            store.update_execution_metadata(execution.execution_id, DeploymentMetadataUpdateRequest(display_name=name))
            updates.append(name)
        finally:
            updating = False

    event.listen(Session, "loaded_as_persistent", concurrent_update)
    try:
        ready = _execution(execution.execution_id, DeploymentStatus.READY, "serving")
        if conflicts == 3:
            with pytest.raises(ConcurrentUpdateError):
                store.save_execution(ready)
        else:
            saved = store.save_execution(ready)
            assert saved.status == DeploymentStatus.READY
            assert saved.metadata.display_name == updates[-1]
    finally:
        event.remove(Session, "loaded_as_persistent", concurrent_update)
    persisted = store.get_execution(execution.execution_id)
    assert len(updates) == conflicts
    assert persisted.metadata.display_name == updates[-1]
    assert persisted.status == (DeploymentStatus.DEPLOYING if conflicts == 3 else DeploymentStatus.READY)


def test_execution_provenance_patch_keeps_cluster_session(tmp_path) -> None:
    store = JsonDeploymentRunStore(tmp_path)
    execution = _execution("execution-patch", DeploymentStatus.READY, "serving")
    store.save_execution(execution)
    store.update_execution_provenance(execution.execution_id, {"runtime_verification": {"status": "pass"}})
    provenance = store.get_execution(execution.execution_id).provenance
    assert provenance["cluster_session_id"] == "session-1"
    assert provenance["runtime_verification"] == {"status": "pass"}


def test_execution_save_does_not_retry_other_persistence_errors(tmp_path) -> None:
    """Do not disguise unrelated database failures as transient version conflicts."""
    store = JsonDeploymentRunStore(tmp_path)
    execution = _execution("execution-error", DeploymentStatus.DEPLOYING, "serving")
    store.save_execution(execution)
    attempts = []

    def fail_read(session: Session, row: object) -> None:
        if getattr(row, "__tablename__", None) == "deployment_evidences":
            attempts.append(row)
            raise RuntimeError("database unavailable")

    event.listen(Session, "loaded_as_persistent", fail_read)
    try:
        with pytest.raises(RuntimeError, match="database unavailable"):
            store.save_execution(execution)
    finally:
        event.remove(Session, "loaded_as_persistent", fail_read)
    assert len(attempts) == 1
    assert store.get_execution(execution.execution_id).status == DeploymentStatus.DEPLOYING


@pytest.mark.parametrize("conflicts", [1, 2, 3])
def test_run_save_reloads_after_concurrent_owner_update(tmp_path, conflicts: int) -> None:
    """Run/job writes retry as a unit without replacing concurrently assigned ownership."""
    store = JsonDeploymentRunStore(tmp_path)
    case = _case("run-race", 0, "execution-race", DeploymentCaseStatus.DEPLOYING)
    run = store.create_run(
        DeploymentRun(
            id="run-race",
            status=DeploymentRunStatus.RUNNING,
            cases=[case],
            source_configurations=[
                DeployableConfiguration(
                    type="optimized-baseline", format="helm", provider_ref="optimized-baseline", checksum="hash"
                )
            ],
        )
    )
    owners = [UserDao().create(UserRecord(username=f"owner-{index}")) for index in range(conflicts)]
    updates = []
    updating = False

    def concurrent_update(session: Session, row: object) -> None:
        nonlocal updating
        if updating or getattr(row, "__tablename__", None) != "deployment_batches" or len(updates) >= conflicts:
            return
        updating = True
        try:
            owner = owners[len(updates)]
            DeploymentBatchDao().set_owner(run.id, owner_user_id=owner.id)
            updates.append(owner.id)
        finally:
            updating = False

    run.status = DeploymentRunStatus.SUCCEEDED
    run.cases[0].status = DeploymentCaseStatus.READY
    event.listen(Session, "loaded_as_persistent", concurrent_update)
    try:
        if conflicts == 3:
            with pytest.raises(ConcurrentUpdateError):
                store.save_run(run)
        else:
            store.save_run(run)
    finally:
        event.remove(Session, "loaded_as_persistent", concurrent_update)
    saved = store.get_run(run.id)
    assert len(updates) == conflicts
    assert DeploymentBatchDao().owner_map([run.id])[run.id][0] == owners[-1].id
    assert saved.status == (DeploymentRunStatus.RUNNING if conflicts == 3 else DeploymentRunStatus.SUCCEEDED)
    assert saved.cases[0].status == (DeploymentCaseStatus.DEPLOYING if conflicts == 3 else DeploymentCaseStatus.READY)


@pytest.fixture
def deployment_store(tmp_path, monkeypatch):
    """A real JSON store holding one run with two deployments."""
    store = JsonDeploymentRunStore(tmp_path)
    run = DeploymentRun(
        id="run-1",
        status=DeploymentRunStatus.SUCCEEDED,
        source_configurations=[
            DeployableConfiguration(
                type="optimized-baseline",
                format="helm",
                provider_ref="optimized-baseline",
                checksum="checksum",
            )
        ],
        provenance={"cluster_server_id": "cluster-1"},
    )
    run.cases = [
        _case("run-1", 0, "execution-ready", DeploymentCaseStatus.READY),
        _case("run-1", 1, "execution-cleaned", DeploymentCaseStatus.CLEANED),
    ]
    store.create_run(run)
    store.save_execution(_execution("execution-ready", DeploymentStatus.READY, "qwen-ready"))
    store.save_execution(_execution("execution-cleaned", DeploymentStatus.CLEANED, "qwen-cleaned"))
    monkeypatch.setattr(executions, "_store", store)
    executions._execution_locks.clear()
    monkeypatch.setattr(usage, "_probes", {})
    return store


@pytest.fixture
def worker_factory(monkeypatch, deployment_store):
    """Bind the real run worker to a fake provider service."""
    from llm_d_lens.deploy.worker import DeploymentRunWorker

    created: dict[str, object] = {}

    def bind(service, lifecycle_error=None):
        worker = DeploymentRunWorker(deployment_store, service, lifecycle_error)
        created["worker"] = worker
        created["service"] = service
        monkeypatch.setattr(executions.deployment_run_manager, "worker_for_run", lambda _run_id: worker)
        return worker

    return bind


def _payload(response):
    assert isinstance(response, JSONResponse)
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_list_returns_all_deployments_with_metadata_fields(deployment_store):
    response = await router.list_executions()

    identifiers = [item["execution_id"] for item in response["items"]]
    assert sorted(identifiers) == ["execution-cleaned", "execution-ready"]
    for item in response["items"]:
        assert item["display_name"] == ""
        assert item["description"] == ""
        assert item["run_id"] == "run-1"
        assert item["case_id"]


@pytest.mark.asyncio
async def test_execution_detail_returns_exact_case_source_configuration(deployment_store, monkeypatch):
    monkeypatch.setattr(router, "_store", deployment_store)
    payload = await router.get_execution("execution-ready")
    run = deployment_store.get_run("run-1")
    assert payload["configuration"] == run.source_configurations[0].model_dump(mode="json")
    assert payload["run_id"] == "run-1"
    assert payload["case_id"] == run.cases[0].id


@pytest.mark.asyncio
async def test_list_filters_by_query_across_metadata_and_serving_facts(deployment_store):
    await executions.update_execution_metadata(
        "execution-ready", DeploymentMetadataUpdateRequest(display_name="Nightly target")
    )

    by_display_name = await router.list_executions(query="nightly")
    by_namespace = await router.list_executions(query="qwen-cleaned")
    by_model = await router.list_executions(query="qwen3-0.6b")

    assert [item["execution_id"] for item in by_display_name["items"]] == ["execution-ready"]
    assert [item["execution_id"] for item in by_namespace["items"]] == ["execution-cleaned"]
    assert len(by_model["items"]) == 2


@pytest.mark.asyncio
async def test_list_filters_by_multiple_statuses_and_limit(deployment_store):
    both = await router.list_executions(statuses=[DeploymentStatus.READY, DeploymentStatus.CLEANED])
    ready_only = await router.list_executions(statuses=[DeploymentStatus.READY])
    limited = await router.list_executions(limit=1)

    assert len(both["items"]) == 2
    assert [item["execution_id"] for item in ready_only["items"]] == ["execution-ready"]
    assert len(limited["items"]) == 1


@pytest.mark.asyncio
async def test_plan_returns_configuration_and_redacted_manifest(deployment_store, tmp_path):
    manifest = tmp_path / "rendered" / "manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text(
        "apiVersion: v1\nkind: Secret\nmetadata:\n  name: token\nstringData:\n  HF_TOKEN: hf_secret\n"
        "---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: vllm\n"
    )
    execution = deployment_store.get_execution("execution-ready")
    execution.artifact.manifest_ref = str(manifest.parent)
    deployment_store.save_execution(execution)

    plan = await router.get_execution_plan("execution-ready")

    assert plan["namespace"] == "qwen-ready"
    assert json.loads(plan["configuration_artifacts"][0]["content"])["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert "hf_secret" not in plan["manifest"]
    assert "HF_TOKEN: <redacted>" in plan["manifest"]
    assert "kind: Deployment" in plan["manifest"]
    assert (await router.get_execution_plan("execution-cleaned"))["manifest"] is None


def test_plan_redaction_covers_list_items_embedded_manifests_and_parse_failures():
    from llm_d_lens.deploy import executions

    listed = "kind: List\nitems:\n- kind: Secret\n  data:\n    token: c2VjcmV0\n- kind: Service\n"
    redacted = executions._redact_manifest_secrets(listed)
    assert "c2VjcmV0" not in redacted and "token: <redacted>" in redacted
    assert executions._redact_manifest_secrets("items: [broken") is None

    manifest = "kind: Secret\nstringData:\n  HF_TOKEN: hf_secret\n"
    combined = executions._redact_manifest_secrets(manifest + "---\n" + listed)
    assert "hf_secret" not in combined and "c2VjcmV0" not in combined
    assert combined.count("<redacted>") == 2
    content = json.dumps({"officialGuide": {"renderedManifest": manifest}, "model": {"name": "m"}})
    result = json.loads(executions._redact_artifact_content(content))
    assert "hf_secret" not in result["officialGuide"]["renderedManifest"]
    assert result["model"] == {"name": "m"}
    assert executions._redact_artifact_content('{"model": "plain"}') == '{"model": "plain"}'


@pytest.mark.asyncio
async def test_patch_updates_only_supplied_fields(deployment_store):
    await router.patch_execution(
        "execution-ready",
        DeploymentMetadataUpdateRequest(display_name="  Qwen production  ", description="First"),
    )
    response = await router.patch_execution("execution-ready", DeploymentMetadataUpdateRequest(description="Second"))

    assert response["display_name"] == "Qwen production"
    assert response["description"] == "Second"
    assert deployment_store.get_execution("execution-ready").metadata.display_name == "Qwen production"


@pytest.mark.asyncio
async def test_patch_can_clear_a_field(deployment_store):
    await router.patch_execution("execution-ready", DeploymentMetadataUpdateRequest(description="temporary"))
    response = await router.patch_execution("execution-ready", DeploymentMetadataUpdateRequest(description=""))

    assert response["description"] == ""


@pytest.mark.asyncio
async def test_patch_never_changes_runtime_facts(deployment_store):
    before = deployment_store.get_execution("execution-ready")

    response = await router.patch_execution("execution-ready", DeploymentMetadataUpdateRequest(display_name="renamed"))
    after = deployment_store.get_execution("execution-ready")

    assert response["name"] == "qwen-0"
    assert response["namespace"] == "qwen-ready"
    assert response["status"] == "ready"
    assert after.namespace == before.namespace
    assert after.status == before.status
    assert after.endpoint == before.endpoint
    assert after.provenance == before.provenance


def test_patch_request_rejects_unknown_and_empty_payloads():
    with pytest.raises(ValueError):
        DeploymentMetadataUpdateRequest()
    with pytest.raises(ValueError):
        DeploymentMetadataUpdateRequest(namespace="other")
    with pytest.raises(ValueError):
        DeploymentMetadataUpdateRequest(display_name="x" * 121)


@pytest.mark.asyncio
async def test_patch_reports_missing_deployment(deployment_store):
    response = await router.patch_execution("absent", DeploymentMetadataUpdateRequest(display_name="x"))

    assert response.status_code == 404
    assert _payload(response)["code"] == "deployment_not_found"


@pytest.mark.asyncio
async def test_metadata_survives_lifecycle_execution_saves(deployment_store):
    await executions.update_execution_metadata(
        "execution-ready", DeploymentMetadataUpdateRequest(description="keep me")
    )

    refreshed = deployment_store.get_execution("execution-ready")
    deployment_store.save_execution(refreshed.model_copy(update={"metadata": DeploymentMetadata()}))

    assert deployment_store.get_execution("execution-ready").metadata.description == "keep me"


@pytest.mark.asyncio
async def test_delete_cleans_resources_then_removes_only_that_record(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker_factory(service)

    await executions.delete_execution("execution-ready")

    assert service.cleaned == ["execution-ready"]
    assert deployment_store.get_execution("execution-ready") is None
    assert deployment_store.get_execution("execution-cleaned") is not None
    run = deployment_store.get_run("run-1")
    assert [case.id for case in run.cases] == ["case-1"]


@pytest.mark.asyncio
async def test_delete_without_namespace_cleanup_keeps_cluster_resources(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker_factory(service)

    await executions.delete_execution("execution-ready", delete_namespace=False)

    assert service.cleaned == []
    assert deployment_store.get_execution("execution-ready") is None


@pytest.mark.asyncio
async def test_delete_of_cleaned_deployment_skips_cluster_cleanup(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker_factory(service)

    await executions.delete_execution("execution-cleaned")

    assert service.cleaned == []
    assert deployment_store.get_execution("execution-cleaned") is None


@pytest.mark.asyncio
async def test_delete_removes_the_run_once_its_last_deployment_is_gone(deployment_store, worker_factory):
    worker_factory(FakeDeploymentService())

    await executions.delete_execution("execution-cleaned")
    await executions.delete_execution("execution-ready")

    assert deployment_store.get_run("run-1") is None


def _resource_bindings(*, role_id):
    """Seed legacy rows directly; public APIs no longer accept deployment shares."""
    service = default_service()
    suffix = uuid4().hex[:8]
    scope_cluster_id = ClusterDao().create(f"dep-{suffix}", "", "apiVersion: v1\nkind: Config").id
    user = UserDao().create(UserRecord(username=f"dep-user-{suffix}"))
    group = GroupDao().create(GroupRecord(name=f"dep-group-{suffix}"))

    def add(subject_type, resource_type, resource_id):
        kwargs = {
            "scope_type": "resource",
            "scope_cluster_id": scope_cluster_id,
            "scope_resource_type": resource_type,
            "scope_resource_id": resource_id,
        }
        if subject_type == "group":
            return service.group_binding_dao.create(
                GroupRoleBindingRecord(group_id=group.id, role_id=role_id, **kwargs)
            )
        return service.user_binding_dao.create(UserRoleBindingRecord(user_id=user.id, role_id=role_id, **kwargs))

    return service, add


@pytest.mark.asyncio
async def test_deleting_a_deployment_revokes_user_and_group_shares(deployment_store, worker_factory):
    worker_factory(FakeDeploymentService())
    role = RoleDao().create(RoleRecord(name="dep-share-role"))
    service, add = _resource_bindings(role_id=role.id)
    add("user", "deployment_execution", "execution-ready")
    add("group", "deployment_execution", "execution-ready")
    add("user", "deployment_execution", "execution-cleaned")
    add("user", "deployment_run", "run-1")

    await executions.delete_execution("execution-ready")

    # The deleted deployment's shares (user and group) are gone.
    assert service.list_resource_shares("deployment_execution", "execution-ready") == []
    # Sibling deployment and the still-existing run keep their shares.
    assert len(service.list_resource_shares("deployment_execution", "execution-cleaned")) == 1
    assert len(service.list_resource_shares("deployment_run", "run-1")) == 1


@pytest.mark.asyncio
async def test_deleting_the_last_deployment_revokes_run_and_execution_shares(deployment_store, worker_factory):
    worker_factory(FakeDeploymentService())
    role = RoleDao().create(RoleRecord(name="dep-share-role"))
    service, add = _resource_bindings(role_id=role.id)
    add("group", "deployment_run", "run-1")
    add("group", "deployment_execution", "execution-ready")

    await executions.delete_execution("execution-cleaned")
    assert len(service.list_resource_shares("deployment_run", "run-1")) == 1

    await executions.delete_execution("execution-ready")

    assert deployment_store.get_run("run-1") is None
    assert service.list_resource_shares("deployment_run", "run-1") == []
    assert service.list_resource_shares("deployment_execution", "execution-ready") == []
    assert deployment_store.list_runs() == []


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_the_deployment_records(deployment_store, worker_factory):
    worker_factory(FakeDeploymentService(fail=True))

    with pytest.raises(CleanFailedError):
        await executions.delete_execution("execution-ready")

    assert deployment_store.get_execution("execution-ready") is not None
    assert len(deployment_store.get_run("run-1").cases) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "clean", "restart", "delete_case", "delete_run"])
@pytest.mark.parametrize("probe_fails", [False, True])
async def test_worker_lifecycle_preserves_borrowed_deployments(
    deployment_store, worker_factory, operation, probe_fails
):
    service = FakeDeploymentService()
    worker = worker_factory(service)
    run = deployment_store.get_run("run-1")
    if operation == "restart":
        run.cases[0].status = DeploymentCaseStatus.FAILED
        deployment_store.save_run(run)

    def probe(_execution_id):
        if probe_fails:
            raise RuntimeError("consumer store unavailable")
        return "active benchmark"

    usage.register_usage_probe("evaluate", probe)
    with pytest.raises(usage.DeploymentInUseError):
        if operation == "delete_run":
            await worker.delete_run("run-1")
        elif operation == "delete_case":
            await worker.delete_case_execution("run-1", "case-0", delete_namespace=False)
        else:
            await getattr(worker, f"{operation}_case")("run-1", "case-0")
    assert service.cleaned == []
    assert deployment_store.get_run("run-1").cases == run.cases
    assert deployment_store.get_execution("execution-ready").status == DeploymentStatus.READY


@pytest.mark.asyncio
async def test_delete_run_preflights_every_case_before_cleanup(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker = worker_factory(service)
    usage.register_usage_probe(
        "evaluate", lambda execution_id: "active benchmark" if execution_id == "execution-cleaned" else None
    )
    with pytest.raises(usage.DeploymentInUseError):
        await worker.delete_run("run-1")
    assert service.cleaned == []
    assert deployment_store.get_execution("execution-ready") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "clean"])
async def test_case_route_reports_native_usage_conflict(deployment_store, worker_factory, monkeypatch, operation):
    worker = worker_factory(FakeDeploymentService())
    monkeypatch.setattr(router, "_store", deployment_store)
    monkeypatch.setattr(router.deployment_run_manager, "worker_for_run", lambda _run_id: worker)
    usage.register_usage_probe("evaluate", lambda _execution_id: "active benchmark")
    response = (
        await router.stop_case("run-1", "case-0")
        if operation == "stop"
        else await router.clean_case("run-1", "case-0", router.CleanCaseRequest())
    )
    assert response.status_code == 409
    assert _payload(response)["code"] == "deployment_in_use"


@pytest.mark.asyncio
async def test_delete_is_refused_when_the_cluster_session_is_gone(deployment_store, worker_factory):
    worker_factory(FakeDeploymentService(), lifecycle_error="cluster session is no longer active")

    response = await router.delete_execution("execution-ready")

    assert response.status_code == 409
    assert _payload(response)["code"] == "deployment_conflict"
    assert deployment_store.get_execution("execution-ready") is not None


@pytest.mark.asyncio
async def test_delete_is_refused_while_a_deployment_is_still_in_use(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker_factory(service)
    usage.register_usage_probe(
        "evaluate", lambda execution_id: "benchmark run bench-1 is running against this deployment"
    )

    response = await router.delete_execution("execution-ready")

    assert response.status_code == 409
    body = _payload(response)
    assert body["code"] == "deployment_in_use"
    assert "bench-1" in body["detail"]
    assert service.cleaned == []
    assert deployment_store.get_execution("execution-ready") is not None


@pytest.mark.asyncio
async def test_delete_is_refused_when_usage_cannot_be_verified(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker_factory(service)

    def broken_probe(_execution_id):
        raise RuntimeError("store unreadable")

    usage.register_usage_probe("evaluate", broken_probe)

    response = await router.delete_execution("execution-ready")

    assert response.status_code == 409
    assert service.cleaned == []


@pytest.mark.asyncio
async def test_delete_cancels_and_cleans_up_a_deployment_still_being_created(deployment_store, worker_factory):
    """Deleting a still-deploying execution cancels the run and proceeds with
    normal cleanup instead of being rejected, so users don't have to
    separately cancel and wait before they can delete it.
    """
    service = FakeDeploymentService()
    worker_factory(service)
    run = deployment_store.get_run("run-1")
    run.status = DeploymentRunStatus.RUNNING
    deployment_store.save_run(run)

    response = await router.delete_execution("execution-ready")

    assert response.status_code == 204
    assert service.cleaned == ["execution-ready"]
    assert deployment_store.get_execution("execution-ready") is None
    remaining_run = deployment_store.get_run("run-1")
    assert remaining_run.status not in {
        DeploymentRunStatus.RUNNING,
        DeploymentRunStatus.QUEUED,
        DeploymentRunStatus.CANCELLING,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [DeploymentRunStatus.QUEUED, DeploymentRunStatus.RUNNING, DeploymentRunStatus.CANCELLING],
)
async def test_delete_cancels_every_in_progress_run_status(deployment_store, worker_factory, status):
    service = FakeDeploymentService()
    worker_factory(service)
    run = deployment_store.get_run("run-1")
    run.status = status
    deployment_store.save_run(run)

    response = await router.delete_execution("execution-ready")

    assert response.status_code == 204
    assert service.cleaned == ["execution-ready"]


@pytest.mark.asyncio
async def test_delete_reports_missing_deployment(deployment_store):
    response = await router.delete_execution("absent")

    assert response.status_code == 404
    assert _payload(response)["code"] == "deployment_not_found"


@pytest.mark.asyncio
async def test_delete_run_self_heals_a_case_whose_execution_record_is_gone(deployment_store, worker_factory):
    """A dangling case.execution_id (record removed out of band, e.g. by a

    prior cleanup that crashed before persisting) must not permanently block
    ``delete_run``: the whole point of deleting the run is to get rid of the
    stale reference, not to require it to still resolve.
    """
    service = FakeDeploymentService()
    worker = worker_factory(service)
    run = deployment_store.get_run("run-1")
    run.cases[0].execution_id = "execution-vanished"
    deployment_store.save_run(run)

    await worker.delete_run("run-1")

    assert service.cleaned == []
    assert deployment_store.get_run("run-1") is None


@pytest.mark.asyncio
async def test_delete_endpoint_can_skip_namespace_cleanup(deployment_store, worker_factory):
    service = FakeDeploymentService()
    worker_factory(service)

    response = await router.delete_execution("execution-ready", delete_namespace=False)

    assert response.status_code == 204
    assert service.cleaned == []
    assert deployment_store.get_execution("execution-ready") is None


def test_store_delete_execution_and_atomic_write_leave_no_temporary_files(tmp_path):
    store = JsonDeploymentRunStore(tmp_path)
    store.save_execution(_execution("execution-a", DeploymentStatus.READY, "namespace-a"))
    store.save_execution(_execution("execution-b", DeploymentStatus.READY, "namespace-b"))

    store.delete_execution("execution-a")

    assert store.get_execution("execution-a") is None
    assert store.get_execution("execution-b") is not None
    assert list((tmp_path / "executions").glob("*.tmp")) == []


def test_store_metadata_update_requires_an_existing_execution(tmp_path):
    from llm_d_lens.deploy.run_store import DeploymentRunStoreError

    store = JsonDeploymentRunStore(tmp_path)

    with pytest.raises(DeploymentRunStoreError):
        store.update_execution_metadata("absent", DeploymentMetadataUpdateRequest(display_name="x"))


def test_legacy_execution_records_load_without_metadata(tmp_path):
    store = JsonDeploymentRunStore(tmp_path)
    execution = _execution("execution-legacy", DeploymentStatus.READY, "namespace")
    store.save_execution(execution.model_copy(update={"metadata": DeploymentMetadata()}))

    loaded = store.get_execution("execution-legacy")

    assert loaded is not None
    assert loaded.metadata == DeploymentMetadata()


@pytest.mark.asyncio
async def test_cancel_evaluation_removes_deployment_records(deployment_store, worker_factory):
    worker = worker_factory(FakeDeploymentService())

    await worker.cancel_evaluation_deployment("run-1")

    assert deployment_store.get_run("run-1") is None
    assert deployment_store.get_execution("execution-ready") is None
    assert deployment_store.get_execution("execution-cleaned") is None
    assert executions.list_execution_contexts() == []


@pytest.mark.asyncio
async def test_cancel_evaluation_retains_records_on_cleanup_failure(deployment_store, worker_factory):
    worker = worker_factory(FakeDeploymentService(fail=True))

    with pytest.raises(ValueError, match="namespace could not be deleted"):
        await worker.cancel_evaluation_deployment("run-1")

    assert deployment_store.get_run("run-1") is not None
    assert deployment_store.get_execution("execution-ready") is not None


def _write_scale_manifest(tmp_path, documents):
    manifest = tmp_path / "scale-manifest.yaml"
    import yaml

    manifest.write_text(yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8")
    return manifest


def _bind_scale_manifest(store, tmp_path, documents):
    execution = store.get_execution("execution-ready")
    manifest = _write_scale_manifest(tmp_path, documents)
    execution.artifact = execution.artifact.model_copy(update={"manifest_ref": str(manifest)})
    store.save_execution(execution)
    return manifest


def _deployment(name, role, replicas):
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": replicas,
            "template": {
                "metadata": {"labels": {"llm-d.ai/role": role}},
                "spec": {"containers": [{"name": "modelserver", "command": ["vllm", "serve"], "args": ["model"]}]},
            },
        },
    }


def _live_get(items):
    return SimpleNamespace(returncode=0, stdout=json.dumps({"items": items}), stderr="")


def _live_items(prefill, decode):
    return [
        {"metadata": {"name": "prefill"}, "spec": {"replicas": prefill}},
        {"metadata": {"name": "decode"}, "spec": {"replicas": decode}},
    ]


@pytest.mark.asyncio
async def test_deployment_scale_plan_reads_live_replica_counts(deployment_store, monkeypatch, tmp_path):
    _bind_scale_manifest(
        deployment_store,
        tmp_path,
        [
            _deployment("prefill", "prefill", 1),
            _deployment("decode", "decode", 1),
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "epp"},
                "spec": {
                    "replicas": 1,
                    "template": {
                        "metadata": {"labels": {}},
                        "spec": {"containers": [{"name": "epp", "image": "router"}]},
                    },
                },
            },
        ],
    )
    calls = []

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        calls.append((command, timeout, cluster_id))
        return _live_get(_live_items(2, 3))

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    plan = await executions.deployment_scale_plan("execution-ready")

    assert plan["prefillReplicas"] == 2
    assert plan["decodeReplicas"] == 3
    assert [item["name"] for item in plan["workloads"]] == ["prefill", "decode"]
    assert calls[0][0] == ["get", "deployments", "--namespace", "qwen-ready", "--output", "json"]


@pytest.mark.asyncio
async def test_scale_deployment_uses_safe_order_and_persists_desired_replicas(deployment_store, monkeypatch, tmp_path):
    _bind_scale_manifest(
        deployment_store,
        tmp_path,
        [_deployment("prefill", "prefill", 1), _deployment("decode", "decode", 1)],
    )
    commands = []

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        commands.append(command)
        if command[0] == "get":
            return _live_get(_live_items(1, 2))
        return SimpleNamespace(returncode=0, stdout="scaled", stderr="")

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    context = await executions.scale_deployment_execution(
        "execution-ready",
        DeploymentScaleRequest(prefill_replicas=2, decode_replicas=3),
        actor="user-1",
    )

    assert [command[2] for command in commands[1:]] == ["prefill", "decode"]
    persisted = deployment_store.get_execution("execution-ready")
    assert persisted.provenance["manual_scale"]["prefillReplicas"] == 2
    assert persisted.provenance["manual_scale"]["decodeReplicas"] == 3
    assert persisted.provenance["manual_scale"]["updatedBy"] == "user-1"
    assert context.scale == {"prefillReplicas": 2, "decodeReplicas": 3, "slices": 1, "manual": True}
    assert context.replicas == 3


@pytest.mark.asyncio
async def test_scale_down_drains_decode_before_prefill(deployment_store, monkeypatch, tmp_path):
    _bind_scale_manifest(
        deployment_store,
        tmp_path,
        [_deployment("prefill", "prefill", 1), _deployment("decode", "decode", 1)],
    )
    commands = []

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        commands.append(command)
        if command[0] == "get":
            return _live_get(_live_items(4, 2))
        return SimpleNamespace(returncode=0, stdout="scaled", stderr="")

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    await executions.scale_deployment_execution(
        "execution-ready",
        DeploymentScaleRequest(prefill_replicas=2, decode_replicas=1),
    )

    assert [command[2] for command in commands[1:]] == ["decode", "prefill"]


@pytest.mark.asyncio
async def test_scale_rejects_unsupported_workloads(deployment_store, monkeypatch, tmp_path):
    _bind_scale_manifest(
        deployment_store,
        tmp_path,
        [
            {
                "apiVersion": "leaderworkerset.x-k8s.io/v1",
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "lws"},
                "spec": {
                    "replicas": 1,
                    "leaderWorkerTemplate": {"workerTemplate": {"spec": {"containers": [{"name": "modelserver"}]}}},
                },
            }
        ],
    )

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        return _live_get([])

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    with pytest.raises(executions.DeploymentScaleError) as failure:
        await executions.deployment_scale_plan("execution-ready")
    assert failure.value.code == "deployment_scale_unsupported"
    assert "unsupported workloads" in str(failure.value)


@pytest.mark.asyncio
async def test_scale_rejects_prefill_for_aggregated_deployment(deployment_store, monkeypatch, tmp_path):
    _bind_scale_manifest(deployment_store, tmp_path, [_deployment("decode", "decode", 1)])

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        return _live_get([{"metadata": {"name": "decode"}, "spec": {"replicas": 1}}])

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    with pytest.raises(executions.DeploymentScaleError) as failure:
        await executions.scale_deployment_execution("execution-ready", DeploymentScaleRequest(prefill_replicas=2))
    assert failure.value.code == "deployment_scale_unsupported"


@pytest.mark.asyncio
async def test_scale_route_returns_updated_execution(deployment_store, monkeypatch, tmp_path):
    _bind_scale_manifest(
        deployment_store,
        tmp_path,
        [_deployment("prefill", "prefill", 1), _deployment("decode", "decode", 1)],
    )

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        if command[0] == "get":
            return _live_get(_live_items(1, 1))
        return SimpleNamespace(returncode=0, stdout="scaled", stderr="")

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    response = await router.scale_execution("execution-ready", DeploymentScaleRequest(decode_replicas=2))

    assert response["scale"]["decodeReplicas"] == 2
    assert response["replicas"] == 2


def _disaggregated_set_document(*, slices=2, prefill=4, decode=1, external_prefill=False, external_decode=False):
    def role(name, replicas):
        value = {
            "name": name,
            "spec": {
                "replicas": replicas,
                "leaderWorkerTemplate": {
                    "workerTemplate": {
                        "spec": {"containers": [{"name": "modelserver", "image": "vllm/vllm-openai:v1"}]}
                    }
                },
            },
        }
        if (name == "prefill" and external_prefill) or (name == "decode" and external_decode):
            value["scaling"] = {"mode": "External"}
        return value

    return {
        "apiVersion": "disaggregatedset.x-k8s.io/v1",
        "kind": "DisaggregatedSet",
        "metadata": {"name": "pd-disagg-xpu"},
        "spec": {"slices": slices, "roles": [role("prefill", prefill), role("decode", decode)]},
    }


@pytest.mark.asyncio
async def test_scale_plan_supports_disaggregated_set(deployment_store, monkeypatch, tmp_path):
    manifest_document = _disaggregated_set_document(slices=2, prefill=4, decode=1)
    _bind_scale_manifest(deployment_store, tmp_path, [manifest_document])
    live_document = _disaggregated_set_document(slices=2, prefill=5, decode=2)

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        if command[:2] == ["get", "deployments"]:
            return _live_get([])
        if command[:2] == ["get", "disaggregatedset"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(live_document), stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    plan = await executions.deployment_scale_plan("execution-ready")

    assert plan["kind"] == "DisaggregatedSet"
    assert plan["slices"] == 2
    assert plan["prefillReplicas"] == 5
    assert plan["decodeReplicas"] == 2
    assert [workload["role"] for workload in plan["workloads"]] == ["prefill", "decode"]
    assert [workload["roleIndex"] for workload in plan["workloads"]] == [0, 1]


@pytest.mark.asyncio
async def test_scale_disaggregated_set_patches_role_replicas_without_slices(deployment_store, monkeypatch, tmp_path):
    document = _disaggregated_set_document(slices=2, prefill=4, decode=1)
    _bind_scale_manifest(deployment_store, tmp_path, [document])
    commands = []

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        commands.append(command)
        if command[:2] == ["get", "deployments"]:
            return _live_get([])
        if command[:2] == ["get", "disaggregatedset"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(document), stderr="")
        if command[0] == "patch":
            return SimpleNamespace(returncode=0, stdout="patched", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    context = await executions.scale_deployment_execution(
        "execution-ready",
        DeploymentScaleRequest(prefill_replicas=6, decode_replicas=2),
    )

    patch_command = next(command for command in commands if command[0] == "patch")
    operations = json.loads(patch_command[patch_command.index("--patch") + 1])
    assert not any(operation["path"] == "/spec/slices" for operation in operations)
    assert {"op": "add", "path": "/spec/roles/0/spec/replicas", "value": 6} in operations
    assert {"op": "add", "path": "/spec/roles/1/spec/replicas", "value": 2} in operations

    persisted = deployment_store.get_execution("execution-ready")
    manual = persisted.provenance["manual_scale"]
    assert manual["kind"] == "DisaggregatedSet"
    assert manual["slices"] == 2
    assert manual["prefillReplicas"] == 6
    assert manual["decodeReplicas"] == 2
    assert context.scale == {
        "prefillReplicas": 6,
        "decodeReplicas": 2,
        "slices": 2,
        "manual": True,
    }
    assert context.expected_model_pods == 16


@pytest.mark.asyncio
async def test_scale_disaggregated_set_allows_unchanged_external_role(deployment_store, monkeypatch, tmp_path):
    document = _disaggregated_set_document(slices=2, prefill=4, decode=1, external_decode=True)
    _bind_scale_manifest(deployment_store, tmp_path, [document])
    commands = []

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        commands.append(command)
        if command[:2] == ["get", "deployments"]:
            return _live_get([])
        if command[:2] == ["get", "disaggregatedset"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(document), stderr="")
        if command[0] == "patch":
            return SimpleNamespace(returncode=0, stdout="patched", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    await executions.scale_deployment_execution(
        "execution-ready",
        DeploymentScaleRequest(prefill_replicas=6, decode_replicas=1),
    )

    patch_command = next(command for command in commands if command[0] == "patch")
    operations = json.loads(patch_command[patch_command.index("--patch") + 1])
    assert operations == [{"op": "add", "path": "/spec/roles/0/spec/replicas", "value": 6}]


@pytest.mark.asyncio
async def test_scale_preserves_concurrent_status_change(deployment_store, monkeypatch, tmp_path):
    document = _disaggregated_set_document(slices=2, prefill=4, decode=1)
    _bind_scale_manifest(deployment_store, tmp_path, [document])

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        if command[:2] == ["get", "deployments"]:
            return _live_get([])
        if command[:2] == ["get", "disaggregatedset"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(document), stderr="")
        if command[0] == "patch":
            current = deployment_store.get_execution("execution-ready")
            deployment_store.save_execution(current.model_copy(update={"status": DeploymentStatus.CLEANED}))
            return SimpleNamespace(returncode=0, stdout="patched", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    await executions.scale_deployment_execution(
        "execution-ready",
        DeploymentScaleRequest(prefill_replicas=6),
    )

    persisted = deployment_store.get_execution("execution-ready")
    assert persisted.status == DeploymentStatus.CLEANED
    assert persisted.provenance["manual_scale"]["prefillReplicas"] == 6


@pytest.mark.asyncio
async def test_scale_disaggregated_set_rejects_external_role(deployment_store, monkeypatch, tmp_path):
    document = _disaggregated_set_document(external_prefill=True)
    _bind_scale_manifest(deployment_store, tmp_path, [document])

    async def fake_run_kubectl(command, timeout=10, cluster_id=None):
        if command[:2] == ["get", "deployments"]:
            return _live_get([])
        if command[:2] == ["get", "disaggregatedset"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="not found")
        raise AssertionError(command)

    monkeypatch.setattr(executions, "run_kubectl", fake_run_kubectl)

    with pytest.raises(executions.DeploymentScaleError) as failure:
        await executions.scale_deployment_execution("execution-ready", DeploymentScaleRequest(prefill_replicas=6))

    assert failure.value.code == "deployment_scale_unsupported"
    assert "Externally managed roles" in str(failure.value)
