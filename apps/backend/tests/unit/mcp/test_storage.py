from datetime import UTC, datetime

import pytest

from llm_d_lens.mcp.context import ToolContext
from llm_d_lens.mcp.lens_api import LensApiError
from llm_d_lens.mcp.registry import all_tools, call_tool

from .conftest import FakeApi
from .test_serving import backend_route_check  # noqa: F401  (autouse: calls must match real routes)

SCOPES = frozenset({"core", "serving", "storage"})
PREFIX_ENV = "LENS_MCP_STORAGE_PATH_PREFIXES"


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
            "createdAt": datetime.now(UTC).isoformat(),
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
    return ToolContext(api=api, owner=api.owner, scopes=SCOPES)


@pytest.fixture
def lens(api: FakeApi, monkeypatch) -> Storage:
    monkeypatch.delenv(PREFIX_ENV, raising=False)
    return Storage(api)


CREATE = {"cluster_id": "c1", "name": "fresh", "kind": "local-disk", "path": "/mnt/models/new", "capacity_gib": 100}


def test_storage_tools_declare_annotations_from_the_spec():
    expected = {
        "storage_volume_create": (False, False, True, False),
        "storage_volume_get": (True, False, True, False),
        "storage_volume_scan": (False, False, True, False),
    }
    tools = {spec.name: spec for spec in all_tools()}
    for name, flags in expected.items():
        spec = tools[name]
        assert (spec.read_only, spec.destructive, spec.idempotent, spec.open_world) == flags, name
        assert spec.profile == "storage"
        assert spec.description.startswith(("[R]", "[W]", "[S]", "[D]")) and "Side effects:" in spec.description


@pytest.mark.asyncio
async def test_create_refuses_hostpath_by_default_and_says_how_to_enable_it(api, ctx, lens):
    payload, is_error = await call_tool(ctx, "storage_volume_create", CREATE)
    assert is_error and payload["error"]["code"] == "invalid_argument" and payload["error"]["field"] == "path"
    assert PREFIX_ENV in payload["error"]["hint"]
    assert api.called("POST", "/storage/volumes") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/etc/kubernetes", "/mnt/modelsX", "/mnt/models/../etc"])
async def test_create_rejects_paths_outside_the_allowed_prefixes_and_lists_them(api, ctx, lens, monkeypatch, path):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models, /data/cache/")
    payload, is_error = await call_tool(ctx, "storage_volume_create", {**CREATE, "path": path})
    assert is_error and payload["error"]["code"] == "invalid_argument" and payload["error"]["field"] == "path"
    assert "/mnt/models" in payload["error"]["hint"] and "/data/cache" in payload["error"]["hint"]
    assert api.called("POST", "/storage/volumes") == []


@pytest.mark.asyncio
async def test_create_under_an_allowed_prefix_reports_provisioning(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    lens.volume = None
    payload, is_error = await call_tool(ctx, "storage_volume_create", CREATE)
    assert not is_error, payload
    (body,) = api.called("POST", "/storage/volumes")
    assert body["capacity"] == "100Gi" and body["purposes"] == ["model-cache"] and body["readOnly"] is False
    assert body["localDisk"] == {"hostPath": "/mnt/models/new"}
    assert payload["volume_id"] == "storage-1"
    assert payload["status"] == "pending" and payload["capacity_gib"] == 100


@pytest.mark.asyncio
async def test_create_nfs_needs_a_server_and_ignores_the_hostpath_allow_list(api, ctx, lens):
    lens.volume = None
    args = {**CREATE, "kind": "nfs", "path": "/exports/models"}
    payload, is_error = await call_tool(ctx, "storage_volume_create", args)
    assert is_error and "nfs_server" in payload["error"]["message"]
    payload, is_error = await call_tool(ctx, "storage_volume_create", {**args, "nfs_server": "10.0.0.5"})
    assert not is_error, payload
    assert api.called("POST", "/storage/volumes")[0]["nfs"] == {"server": "10.0.0.5", "path": "/exports/models"}


@pytest.mark.asyncio
async def test_create_with_an_existing_name_is_already_exists(api, ctx, lens, monkeypatch):
    monkeypatch.setenv(PREFIX_ENV, "/mnt/models")
    payload, is_error = await call_tool(ctx, "storage_volume_create", {**CREATE, "name": "models"})
    assert is_error and payload["error"]["code"] == "already_exists" and payload["error"]["field"] == "name"
    assert api.called("POST", "/storage/volumes") == []


@pytest.mark.asyncio
async def test_get_reports_capacity_node_coverage_drift_and_managed_models(api, ctx, lens):
    volume_id = "storage-1"
    lens.entries.append(entry("model-cache-2", "hand/copied", ["n1", "n2"]))
    payload, is_error = await call_tool(
        ctx, "storage_volume_get", {"volume_id": volume_id, "include": ["nodes", "models", "k8s_status"]}
    )
    assert not is_error, payload
    assert payload["status"] == "ready" and payload["capacity_gib"] == 500 and payload["used_gib"] == 4
    assert payload["node_drift"] == ["n3"]
    assert {n["node"]: n["status"] for n in payload["nodes"]} == {"n1": "ready", "n2": "ready", "n3": "pending_ack"}
    assert {m["model"] for m in payload["models"]} == {"Qwen/Qwen3-0.6B", "hand/copied"}
    assert payload["models"][0]["nodes_pending"] == ["n3"]
    assert payload["k8s_status"] == [{"volumeId": "storage-1", "pv": "Bound"}]
    assert api.called("GET", "/model-cache/entries") == [{"storage_volume_id": "storage-1"}]


@pytest.mark.asyncio
async def test_get_unknown_volume_is_not_found(ctx, lens):
    missing, is_error = await call_tool(ctx, "storage_volume_get", {"volume_id": "storage-nope"})
    assert is_error and missing["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_scan_reports_models_found_once(api, ctx, lens):
    lens.discoverable = [entry("model-cache-9", "hand/copied", ["n1", "n2"])]
    args = {"volume_id": "storage-1", "wait_seconds": 0}
    payload, is_error = await call_tool(ctx, "storage_volume_scan", args)
    assert not is_error, payload
    assert payload["scan"]["new_models"] == [{"model": "hand/copied", "model_cache_id": "model-cache-9"}]
    again, _ = await call_tool(ctx, "storage_volume_scan", args)
    assert again["scan"]["new_models"] == []


@pytest.mark.asyncio
async def test_scan_without_registering_sends_no_scan(api, ctx, lens):
    payload, is_error = await call_tool(
        ctx, "storage_volume_scan", {"volume_id": "storage-1", "register_models": False, "wait_seconds": 0}
    )
    assert not is_error and api.called("POST", "/storage/volumes/storage-1/scan-models") == []
    assert "register_models=true" in payload["scan"]["note"]


@pytest.mark.asyncio
async def test_scan_requires_a_ready_volume(ctx, lens):
    lens.volume["status"] = "pending"
    payload, is_error = await call_tool(ctx, "storage_volume_scan", {"volume_id": "storage-1", "wait_seconds": 0})
    assert is_error and payload["error"]["code"] == "conflict_state"


@pytest.mark.asyncio
async def test_confirm_nodes_is_refused_until_the_new_node_holds_every_model(api, ctx, lens):
    args = {"volume_id": "storage-1", "confirm_nodes": True, "register_models": False, "wait_seconds": 0}
    payload, is_error = await call_tool(ctx, "storage_volume_scan", args)
    assert not is_error, payload
    confirmation = payload["nodes_confirmation"]
    assert confirmation["confirmed"] is False and confirmation["new_nodes"] == ["n3"]
    assert "n3: model Qwen/Qwen3-0.6B is not synced" in confirmation["problems"][0]
    assert api.called("POST", "/storage/volumes/storage-1/acknowledge-nodes") == []

    lens.entries = [entry("model-cache-1", "Qwen/Qwen3-0.6B", ["n1", "n2", "n3"])]
    lens.nodes[2]["schedulable"] = False
    payload, _ = await call_tool(ctx, "storage_volume_scan", args)
    assert "n3: node is not ready or not schedulable" in payload["nodes_confirmation"]["problems"]

    lens.nodes[2]["schedulable"] = True
    payload, _ = await call_tool(ctx, "storage_volume_scan", args)
    assert payload["nodes_confirmation"] == {"confirmed": True, "new_nodes": ["n3"]}
    assert len(api.called("POST", "/storage/volumes/storage-1/acknowledge-nodes")) == 1
    assert payload["node_drift"] == [] and {n["status"] for n in payload["nodes"]} == {"ready"}


STORAGE = {"kind": "storage_volume", "id": "storage-1", "name": "models"}


@pytest.mark.asyncio
async def test_storage_volume_kind_lists_every_volume_and_blocks_deletion_while_in_use(api, ctx, lens):
    lens.volume["inUseCount"] = 2
    listed, _ = await call_tool(ctx, "resource_list", {"kinds": ["storage_volume"], "owner": "all"})
    (item,) = listed["resources"]
    assert item["status"] == "ready" and item["in_use_count"] == 2 and item["ownership"] == "unrecorded"
    mine, _ = await call_tool(ctx, "resource_list", {"kinds": ["storage_volume"]})
    assert mine["resources"] == []

    result, _ = await call_tool(ctx, "resource_delete", {"items": [STORAGE]})
    assert result["results"][0]["status"] == "blocked" and result["results"][0]["error"] == "in_use"
    assert lens.volume is not None


@pytest.mark.asyncio
async def test_storage_volume_delete_keeps_model_files_and_reports_the_caches_it_closes(api, ctx, lens):
    done, _ = await call_tool(ctx, "resource_delete", {"items": [STORAGE]})
    (result,) = done["results"]
    assert result["status"] == "deleted" and lens.volume is None
    assert result["also_deletes"] == [{"kind": "model_cache", "id": "model-cache-1"}]
    assert api.called("DELETE", "/storage/volumes/storage-1") == [{"keep_model_files": "true"}]


@pytest.mark.asyncio
async def test_storage_volume_delete_refuses_a_stale_name(api, ctx, lens):
    payload, is_error = await call_tool(ctx, "resource_delete", {"items": [{**STORAGE, "name": "something-else"}]})
    assert is_error and payload["error"]["code"] == "id_mismatch" and lens.volume is not None
