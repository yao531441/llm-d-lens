import pytest

from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.ledger import Ledger
from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.registry import all_tools, call_tool

from .conftest import FakeApi
from .test_serving import backend_route_check  # noqa: F401  (autouse: calls must match real routes)

SCOPES = frozenset({"core", "serving", "storage"})
PREFIX_ENV = "LENS_MCP_V2_STORAGE_PATH_PREFIXES"


def entry(entry_id: str, repo: str, ready_nodes: list[str], pending: list[str] = ()) -> dict:
    return {
        "id": entry_id,
        "status": "ready",
        "source": {"huggingface": {"repoId": repo, "revision": "main"}},
        "sizeBytes": 2 * 2**30,
        "nodeProgress": [{"node": n, "status": "ready"} for n in ready_nodes]
        + [{"node": n, "status": "syncing"} for n in pending],
        "pendingSyncNodes": list(pending),
    }


class Storage:
    """A tiny stateful stand-in for the Lens storage and model-cache APIs."""

    def __init__(self, api: FakeApi):
        self.api = api
        self.template = {
            "id": "storage-1",
            "clusterId": "c1",
            "name": "models",
            "kind": "local-disk",
            "status": "ready",
            "capacity": "500Gi",
            "usedBytes": 4 * 2**30,
            "readOnly": False,
            "purposes": ["model-cache"],
            "localDisk": {"hostPath": "/mnt/models"},
            "knownNodes": ["n1", "n2"],
            "nodesAdded": ["n3"],
            "inUseCount": 0,
        }
        self.volume = dict(self.template)
        self.nodes = [{"name": n, "ready": True, "schedulable": True} for n in ("n1", "n2", "n3")]
        self.entries = [entry("model-cache-1", "Qwen/Qwen3-0.6B", ["n1", "n2"], ["n3"])]
        self.discoverable: list[dict] = []
        api.on("GET", "/storage/volumes", lambda p, b: {"items": [self.volume] if self.volume else []})
        api.on("POST", "/storage/volumes", self.create)
        api.on("GET", "/storage/volumes/storage-1", lambda p, b: self.volume)
        api.on("GET", "/storage/nodes", lambda p, b: {"items": self.nodes})
        api.on("GET", "/model-cache/entries", lambda p, b: {"items": self.entries})
        api.on("POST", "/storage/volumes/storage-1/scan-models", self.scan)
        api.on("POST", "/storage/volumes/storage-1/acknowledge-nodes", self.acknowledge)
        api.on("DELETE", "/storage/volumes/storage-1", self.delete)
        api.on("GET", "/storage/volume-resource-status", {"items": [{"volumeId": "storage-1", "pv": "Bound"}]})

    def create(self, params, body):
        self.volume = {
            **self.template,
            "id": "storage-1",
            "name": body["name"],
            "kind": body["kind"],
            "status": "pending",
            "capacity": body["capacity"],
            "localDisk": body.get("localDisk"),
            "nfs": body.get("nfs"),
            "knownNodes": [],
            "nodesAdded": [],
        }
        return self.volume

    def scan(self, params, body):
        self.entries += self.discoverable
        self.discoverable = []
        return {"status": "scanning"}

    def acknowledge(self, params, body):
        self.volume["knownNodes"] = [n["name"] for n in self.nodes]
        self.volume["nodesAdded"] = []
        return self.volume

    def delete(self, params, body):
        if self.volume["inUseCount"]:
            raise LensApiError(409, {"code": "storage_volume_in_use", "detail": "in use"}, "DELETE", "")
        self.volume = None


@pytest.fixture
def ctx(api: FakeApi) -> ToolContext:
    return ToolContext(api=api, ledger=Ledger(":memory:"), owner=api.owner, scopes=SCOPES)


@pytest.fixture
def lens(api: FakeApi, monkeypatch) -> Storage:
    monkeypatch.delenv(PREFIX_ENV, raising=False)
    return Storage(api)


def own_volume(ctx: ToolContext, owner: str | None = None) -> str:
    return ctx.ledger.add_resource(
        kind="storage_volume", owner=owner or ctx.owner, cluster_id="c1", name="models", lens_execution_id="storage-1"
    ).id


CREATE = {"cluster_id": "c1", "name": "fresh", "kind": "local-disk", "path": "/mnt/models/new", "capacity_gib": 100}


def test_storage_tools_declare_annotations_from_the_spec():
    expected = {
        "create_storage_volume": (False, False, True, False),
        "get_storage_volume": (True, False, True, False),
        "scan_storage_volume": (False, False, True, False),
    }
    tools = {spec.name: spec for spec in all_tools()}
    for name, flags in expected.items():
        spec = tools[name]
        assert (spec.read_only, spec.destructive, spec.idempotent, spec.open_world) == flags, name
        assert spec.profile == "storage"
        assert spec.description.startswith(("[R]", "[W]", "[S]", "[D]")) and "Side effects:" in spec.description


@pytest.mark.asyncio
async def test_create_refuses_hostpath_by_default_and_says_how_to_enable_it(api, ctx, lens):
    payload, is_error = await call_tool(ctx, "create_storage_volume", CREATE)
    assert is_error and payload["error"]["code"] == "invalid_argument" and payload["error"]["field"] == "path"
    assert PREFIX_ENV in payload["error"]["hint"]
    assert api.called("POST", "/storage/volumes") == [] and ctx.ledger.list_resources(owner="user-a") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/etc/kubernetes", "/mnt/modelsX", "/mnt/models/../etc"])
async def test_create_rejects_paths_outside_the_allowed_prefixes_and_lists_them(api, ctx, lens, monkeypatch, path):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models, /data/cache/")
    payload, is_error = await call_tool(ctx, "create_storage_volume", {**CREATE, "path": path})
    assert is_error and payload["error"]["code"] == "invalid_argument" and payload["error"]["field"] == "path"
    assert "/mnt/models" in payload["error"]["hint"] and "/data/cache" in payload["error"]["hint"]
    assert api.called("POST", "/storage/volumes") == []


@pytest.mark.asyncio
async def test_create_under_an_allowed_prefix_records_the_ledger_and_reports_provisioning(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    lens.volume = None
    payload, is_error = await call_tool(ctx, "create_storage_volume", CREATE)
    assert not is_error, payload
    (body,) = api.called("POST", "/storage/volumes")
    assert body["capacity"] == "100Gi" and body["purposes"] == ["model-cache"] and body["readOnly"] is False
    assert body["localDisk"] == {"hostPath": "/mnt/models/new"}
    row = ctx.ledger.get(payload["volume_id"])
    assert row.kind == "storage_volume" and row.state == "active" and row.lens_execution_id == "storage-1"
    assert payload["status"] == "pending" and payload["capacity_gib"] == 100 and payload["is_mine"]


@pytest.mark.asyncio
async def test_create_nfs_needs_a_server_and_ignores_the_hostpath_allow_list(api, ctx, lens):
    lens.volume = None
    args = {**CREATE, "kind": "nfs", "path": "/exports/models"}
    payload, is_error = await call_tool(ctx, "create_storage_volume", args)
    assert is_error and "nfs_server" in payload["error"]["message"]
    payload, is_error = await call_tool(ctx, "create_storage_volume", {**args, "nfs_server": "10.0.0.5"})
    assert not is_error, payload
    assert api.called("POST", "/storage/volumes")[0]["nfs"] == {"server": "10.0.0.5", "path": "/exports/models"}


@pytest.mark.asyncio
async def test_create_with_an_existing_name_is_already_exists(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    payload, is_error = await call_tool(ctx, "create_storage_volume", {**CREATE, "name": "models"})
    assert is_error and payload["error"]["code"] == "already_exists" and payload["error"]["field"] == "name"
    assert api.called("POST", "/storage/volumes") == []


@pytest.mark.asyncio
async def test_get_reports_capacity_node_coverage_drift_and_managed_models(api, ctx, lens):
    volume_id = own_volume(ctx)
    ctx.ledger.add_resource(
        kind="model_cache", owner="user-a", cluster_id="c1", name="Qwen", lens_execution_id="model-cache-1"
    )
    lens.entries.append(entry("model-cache-2", "hand/copied", ["n1", "n2"]))
    payload, is_error = await call_tool(
        ctx, "get_storage_volume", {"volume_id": volume_id, "include": ["nodes", "models", "k8s_status"]}
    )
    assert not is_error, payload
    assert payload["status"] == "ready" and payload["capacity_gib"] == 500 and payload["used_gib"] == 4
    assert payload["node_drift"] == ["n3"]
    assert {n["node"]: n["status"] for n in payload["nodes"]} == {"n1": "ready", "n2": "ready", "n3": "pending_ack"}
    managed = {m["model"]: m["managed_by_lens"] for m in payload["models"]}
    assert managed == {"Qwen/Qwen3-0.6B": True, "hand/copied": False}
    assert payload["models"][0]["nodes_pending"] == ["n3"]
    assert payload["k8s_status"] == [{"volumeId": "storage-1", "pv": "Bound"}]
    assert api.called("GET", "/model-cache/entries") == [{"storage_volume_id": "storage-1"}]


@pytest.mark.asyncio
async def test_get_shows_volumes_of_others_read_only(ctx, lens):
    payload, is_error = await call_tool(ctx, "get_storage_volume", {"volume_id": "storage-1"})
    assert not is_error and payload["is_mine"] is False and payload["volume_id"] == "storage-1"
    missing, is_error = await call_tool(ctx, "get_storage_volume", {"volume_id": "vol-nope"})
    assert is_error and missing["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_scan_registers_new_models_in_the_ledger_once(api, ctx, lens):
    volume_id = own_volume(ctx)
    lens.discoverable = [entry("model-cache-9", "hand/copied", ["n1", "n2"])]
    args = {"volume_id": volume_id, "wait_seconds": 0}
    payload, is_error = await call_tool(ctx, "scan_storage_volume", args)
    assert not is_error, payload
    assert [m["model"] for m in payload["scan"]["new_models"]] == ["hand/copied"]
    (cache,) = [r for r in ctx.ledger.list_resources(owner="user-a", kind="model_cache") if r.name == "hand/copied"]
    assert cache.lens_execution_id == "model-cache-9" and cache.meta["storage_volume_id"] == "storage-1"
    assert any(m["managed_by_lens"] for m in payload["models"] if m["model"] == "hand/copied")
    again, _ = await call_tool(ctx, "scan_storage_volume", args)
    assert again["scan"]["new_models"] == []


@pytest.mark.asyncio
async def test_scan_without_registering_sends_no_scan(api, ctx, lens):
    volume_id = own_volume(ctx)
    payload, is_error = await call_tool(
        ctx, "scan_storage_volume", {"volume_id": volume_id, "register_models": False, "wait_seconds": 0}
    )
    assert not is_error and api.called("POST", "/storage/volumes/storage-1/scan-models") == []
    assert "register_models=true" in payload["scan"]["note"]


@pytest.mark.asyncio
async def test_scan_requires_ownership_and_a_ready_volume(ctx, lens):
    payload, is_error = await call_tool(ctx, "scan_storage_volume", {"volume_id": "storage-1", "wait_seconds": 0})
    assert is_error and payload["error"]["code"] == "owner_unknown"
    other = own_volume(ctx, owner="bob")
    payload, is_error = await call_tool(ctx, "scan_storage_volume", {"volume_id": other, "wait_seconds": 0})
    assert is_error and payload["error"]["code"] == "not_owner"
    mine = ctx.ledger.add_resource(
        kind="storage_volume", owner="user-a", cluster_id="c1", name="m2", lens_execution_id="storage-1"
    ).id
    lens.volume["status"] = "pending"
    payload, is_error = await call_tool(ctx, "scan_storage_volume", {"volume_id": mine, "wait_seconds": 0})
    assert is_error and payload["error"]["code"] == "conflict_state"


@pytest.mark.asyncio
async def test_confirm_nodes_is_refused_until_the_new_node_holds_every_model(api, ctx, lens):
    volume_id = own_volume(ctx)
    args = {"volume_id": volume_id, "confirm_nodes": True, "register_models": False, "wait_seconds": 0}
    payload, is_error = await call_tool(ctx, "scan_storage_volume", args)
    assert not is_error, payload
    confirmation = payload["nodes_confirmation"]
    assert confirmation["confirmed"] is False and confirmation["new_nodes"] == ["n3"]
    assert "n3: model Qwen/Qwen3-0.6B is not synced" in confirmation["problems"][0]
    assert api.called("POST", "/storage/volumes/storage-1/acknowledge-nodes") == []

    lens.entries = [entry("model-cache-1", "Qwen/Qwen3-0.6B", ["n1", "n2", "n3"])]
    lens.nodes[2]["schedulable"] = False
    payload, _ = await call_tool(ctx, "scan_storage_volume", args)
    assert "n3: node is not ready or not schedulable" in payload["nodes_confirmation"]["problems"]

    lens.nodes[2]["schedulable"] = True
    payload, _ = await call_tool(ctx, "scan_storage_volume", args)
    assert payload["nodes_confirmation"] == {"confirmed": True, "new_nodes": ["n3"]}
    assert len(api.called("POST", "/storage/volumes/storage-1/acknowledge-nodes")) == 1
    assert payload["node_drift"] == [] and {n["status"] for n in payload["nodes"]} == {"ready"}


@pytest.mark.asyncio
async def test_storage_volume_kind_lists_dependents_and_blocks_deletion(api, ctx, lens):
    volume_id = own_volume(ctx)
    lens.volume["inUseCount"] = 2
    listed, _ = await call_tool(ctx, "list_resources", {"kinds": ["storage_volume"], "include": ["dependents"]})
    (item,) = listed["resources"]
    assert item["live_status"] == "ready" and item["in_use_count"] == 2
    assert {d["kind"] for d in item["dependents"]} == {"deployment", "model_cache"}

    result, _ = await call_tool(ctx, "delete_resources", {"ids": [volume_id], "cascade": True})
    assert result["results"][0]["status"] == "blocked" and result["results"][0]["error"] == "in_use"
    assert ctx.ledger.get(volume_id).state == "active"


@pytest.mark.asyncio
async def test_storage_volume_delete_keeps_model_files_and_closes_scan_registered_caches(api, ctx, lens):
    volume_id = own_volume(ctx)
    cache = ctx.ledger.add_resource(
        kind="model_cache",
        owner="user-a",
        cluster_id="c1",
        name="hand/copied",
        lens_execution_id="model-cache-9",
        meta={"storage_volume_id": "storage-1"},
    )
    blocked, _ = await call_tool(ctx, "delete_resources", {"ids": [volume_id]})
    assert blocked["results"][0]["status"] == "blocked" and lens.volume is not None
    done, _ = await call_tool(ctx, "delete_resources", {"ids": [volume_id], "cascade": True})
    assert done["deleted"] == [volume_id] and lens.volume is None
    assert api.called("DELETE", "/storage/volumes/storage-1") == [{"keep_model_files": "true"}]
    assert ctx.ledger.get(cache.id).state == "deleted"


@pytest.mark.asyncio
async def test_list_others_shows_foreign_volumes_read_only(ctx, lens):
    own_volume(ctx, owner="bob")
    listed, _ = await call_tool(ctx, "list_resources", {"kinds": ["storage_volume"], "owner": "all"})
    assert [(e["name"], e["ownership"], e["lens_id"]) for e in listed["resources"]] == [
        ("models", "shared", "storage-1")
    ]
    lens.volume["id"] = "storage-1"
    fresh = ToolContext(api=ctx.api, ledger=Ledger(":memory:"), owner="user-a", scopes=SCOPES)
    listed, _ = await call_tool(fresh, "list_resources", {"kinds": ["storage_volume"], "owner": "all"})
    assert [e["ownership"] for e in listed["resources"]] == ["unknown"]


@pytest.mark.asyncio
async def test_timeout_during_create_keeps_the_row_and_get_reconciles_by_name(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    lens.volume = None

    def timeout(params, body):
        lens.create(params, body)  # Lens did create it
        raise TimeoutError

    api.on("POST", "/storage/volumes", timeout)
    payload, is_error = await call_tool(ctx, "create_storage_volume", CREATE)
    assert is_error and payload["error"]["code"] == "upstream_unavailable"
    volume_id = payload["error"]["volume_id"]
    assert ctx.ledger.get(volume_id).state == "creating"

    got, is_error = await call_tool(ctx, "get_storage_volume", {"volume_id": volume_id})
    assert not is_error and got["is_mine"] and got["lens_id"] == "storage-1"
    row = ctx.ledger.get(volume_id)
    assert row.state == "active" and row.lens_execution_id == "storage-1"


@pytest.mark.asyncio
async def test_retry_after_unknown_outcome_points_at_the_existing_volume(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    lens.volume = None
    row = ctx.ledger.add_resource(
        kind="storage_volume", owner="user-a", cluster_id="c1", name="fresh", state="creating"
    )
    lens.create(
        None, {"name": "fresh", "kind": "local-disk", "capacity": "100Gi", "localDisk": {"hostPath": "/mnt/models/new"}}
    )
    payload, is_error = await call_tool(ctx, "create_storage_volume", CREATE)
    assert is_error and payload["error"]["code"] == "already_exists" and payload["error"]["volume_id"] == row.id
    assert ctx.ledger.get(row.id).state == "active"


@pytest.mark.asyncio
async def test_create_discards_the_row_only_on_a_clear_client_error(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    lens.volume = None
    api.on("POST", "/storage/volumes", LensApiError(503, {"detail": "down"}, "POST", ""))
    payload, is_error = await call_tool(ctx, "create_storage_volume", CREATE)
    assert is_error and payload["error"]["code"] == "upstream_unavailable"
    assert [r.state for r in ctx.ledger.list_resources(owner="user-a", kind="storage_volume")] == ["creating"]
    api.on("POST", "/storage/volumes", LensApiError(422, {"detail": "bad"}, "POST", ""))
    payload, is_error = await call_tool(ctx, "create_storage_volume", {**CREATE, "name": "other"})
    assert is_error and payload["error"]["code"] != "upstream_unavailable"
    assert {r.name: r.state for r in ctx.ledger.list_resources(owner="user-a", kind="storage_volume")} == {
        "fresh": "creating"
    }
