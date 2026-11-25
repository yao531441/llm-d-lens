"""Tests for PCIe-root topology and placement on recorded ResourceSlice shapes."""

from __future__ import annotations

import importlib
import json

import pytest

from llm_d_lens.cluster import registry, service
from llm_d_lens.cluster.errors import ClusterOverviewError
from llm_d_lens.hardware.registry import get_profile
from llm_d_lens.infrastructure.process import CommandResult

ROOT = "resource.kubernetes.io/pcieRoot"


def _node(name: str, *, ready: bool = True, unschedulable: bool = False) -> dict:
    return {
        "metadata": {"name": name},
        "spec": {"unschedulable": unschedulable},
        "status": {"conditions": [{"type": "Ready", "status": "True" if ready else "False"}]},
    }


def _gpu_slice(node: str, roots: dict[str, list[str]]) -> dict:
    devices = [{"name": gpu, "attributes": {ROOT: {"string": root}}} for root, gpus in roots.items() for gpu in gpus]
    return {"spec": {"driver": "gpu.intel.com", "nodeName": node, "pool": {"name": node}, "devices": devices}}


def _nic_slice(node: str, roots: dict[str, str], *, rdma: bool = True) -> dict:
    devices = [
        {
            "name": nic,
            "attributes": {ROOT: {"string": root}, "dra.net/rdma": {"bool": rdma}, "dra.net/numaNode": {"int": 1}},
        }
        for root, nic in roots.items()
    ]
    return {"spec": {"driver": "dra.net", "nodeName": node, "pool": {"name": node}, "devices": devices}}


def _claim(node: str, *devices: str, admin: bool = False) -> dict:
    results = [
        {
            "pool": node,
            "device": device,
            "driver": "dra.net" if "nic" in device else "gpu.intel.com",
            **({"adminAccess": True} if admin else {}),
        }
        for device in devices
    ]
    return {"status": {"allocation": {"devices": {"results": results}}}}


def _cluster_nodes(*names: str, roots: int = 4) -> tuple[list[dict], list[dict]]:
    """Each node: ``roots`` PCIe roots with 2 GPUs and 1 RDMA NIC per root."""
    nodes, slices = [], []
    for name in names:
        nodes.append(_node(name))
        slices.append(
            _gpu_slice(name, {f"pci0000:{i:02x}": [f"{name}-g{2 * i}", f"{name}-g{2 * i + 1}"] for i in range(roots)})
        )
        slices.append(_nic_slice(name, {f"pci0000:{i:02x}": f"{name}-nic{i}" for i in range(roots)}))
    return nodes, slices


def _topology(nodes, slices, claims=()):
    return service.build_topology(nodes, slices, list(claims), get_profile("intel-xpu"))


def _roles(**tp_and_replicas: tuple[int, int]) -> list[dict]:
    return [
        {"name": name, "replicas": replicas, "tensorParallelSize": tp}
        for name, (replicas, tp) in tp_and_replicas.items()
    ]


def test_topology_groups_devices_into_roots_and_marks_free_roots():
    nodes, slices = _cluster_nodes("n1")
    topology = _topology(nodes, slices)

    assert topology["gpusPerRoot"] == 2
    assert topology["requiresNic"] is True
    (node,) = topology["nodes"]
    assert node["freeRoots"] == 4
    assert node["roots"][0] == {
        "root": "pci0000:00",
        "gpus": ["n1-g0", "n1-g1"],
        "nics": ["n1-nic0"],
        "numa": 1,
        "busyGpus": 0,
        "free": True,
    }


def test_allocated_gpu_or_missing_nic_makes_a_root_busy_and_monitoring_claims_are_ignored():
    nodes, slices = _cluster_nodes("n1")
    slices[1]["spec"]["devices"].pop(1)  # DRANET unpublishes a NIC once it is allocated
    claims = [_claim("n1", "n1-g0"), _claim("n1", "n1-g4", "n1-g5", admin=True)]
    topology = _topology(nodes, slices, claims)

    free = {root["root"]: root["free"] for root in topology["nodes"][0]["roots"]}
    assert free == {"pci0000:00": False, "pci0000:01": False, "pci0000:02": True, "pci0000:03": True}
    assert topology["nodes"][0]["busyGpus"] == 1


def test_a_root_with_missing_gpus_is_not_counted_as_free_capacity():
    nodes, slices = _cluster_nodes("n1", roots=3)
    slices[0]["spec"]["devices"].pop(2)  # root 01 publishes a single GPU
    topology = _topology(nodes, slices)

    assert topology["gpusPerRoot"] == 2
    assert topology["nodes"][0]["freeRoots"] == 2
    tp6 = service.plan_placement(topology, _roles(serving=(1, 6)))
    assert tp6["fits"] is False


def test_non_rdma_nics_and_nodes_without_roots_are_ignored():
    nodes, slices = _cluster_nodes("n1", roots=1)
    slices.append(_nic_slice("n1", {"pci0000:09": "n1-eth"}, rdma=False))
    nodes.append(_node("control-plane"))
    topology = _topology(nodes, slices)

    assert [node["name"] for node in topology["nodes"]] == ["n1"]
    assert [root["root"] for root in topology["nodes"][0]["roots"]] == ["pci0000:00"]


def test_pd_topology_fits_on_disjoint_nodes_and_prefers_idle_nodes():
    nodes, slices = _cluster_nodes("n1", "n2", "n3")
    topology = _topology(nodes, slices, [_claim("n1", "n1-g0")])

    plan = service.plan_placement(topology, _roles(prefill=(1, 4), decode=(1, 4)), role_nodes_disjoint=True)

    assert plan["fits"] is True
    assert {node for pods in plan["assignment"].values() for node in pods} == {"n2", "n3"}
    assert plan["freeRoots"] == {"n1": 3, "n2": 4, "n3": 4}


def test_disjoint_roles_do_not_fit_when_a_single_node_would_be_needed_for_both():
    nodes, slices = _cluster_nodes("n1")
    topology = _topology(nodes, slices)
    roles = _roles(prefill=(1, 4), decode=(1, 4))

    disjoint = service.plan_placement(topology, roles, role_nodes_disjoint=True)
    shared = service.plan_placement(topology, roles)

    assert disjoint["fits"] is False
    assert disjoint["reasons"] == ["No placement was found for all roles together on disjoint nodes"]
    assert shared["fits"] is True
    assert shared["assignment"] == {"prefill": {"n1": 1}, "decode": {"n1": 1}}


def test_free_gpu_total_is_not_enough_when_roots_are_split_across_nodes():
    nodes, slices = _cluster_nodes("n1", "n2", roots=2)
    topology = _topology(nodes, slices, [_claim("n1", "n1-g0"), _claim("n2", "n2-g2")])

    plan = service.plan_placement(topology, _roles(decode=(1, 4)))

    assert plan["fits"] is False
    assert plan["reasons"] == ["decode: needs 1 pod(s) x 2 whole root(s), but usable nodes can host 0"]
    assert plan["freeRoots"] == {"n1": 1, "n2": 1}


def test_not_ready_unschedulable_and_excluded_nodes_are_unusable():
    nodes, slices = _cluster_nodes("n1", "n2", "n3")
    nodes[0] = _node("n1", ready=False)
    nodes[1] = _node("n2", unschedulable=True)
    topology = _topology(nodes, slices)

    plan = service.plan_placement(topology, _roles(decode=(1, 2)), exclude_nodes=("n3",))

    assert plan["fits"] is False
    assert plan["unusableNodes"] == ["n1", "n2", "n3"]


def test_tensor_parallel_size_must_be_whole_roots():
    nodes, slices = _cluster_nodes("n1")
    plan = service.plan_placement(_topology(nodes, slices), _roles(decode=(1, 3)))

    assert plan["fits"] is False
    assert plan["reasons"] == ["decode: tensorParallelSize=3 is not a multiple of 2 GPUs per root"]


def test_replicas_spread_over_nodes_when_one_node_is_not_enough():
    nodes, slices = _cluster_nodes("n1", "n2")
    plan = service.plan_placement(_topology(nodes, slices), _roles(decode=(3, 4)))

    assert plan["fits"] is True
    assert sorted(plan["assignment"]["decode"].values()) == [1, 2]


@pytest.mark.asyncio
async def test_probe_reads_nodes_slices_and_claims_through_kubectl(monkeypatch):
    nodes, slices = _cluster_nodes("n1")
    payloads = {"nodes": nodes, "resourceslices.resource.k8s.io": slices, "resourceclaims.resource.k8s.io": []}

    async def fake_run_kubectl(argv, **_kwargs):
        return CommandResult(argv=tuple(argv), returncode=0, stdout=json.dumps({"items": payloads[argv[1]]}), stderr="")

    monkeypatch.setattr(service, "run_kubectl", fake_run_kubectl)
    cluster = registry.Cluster(id="c1", name="c", description="", created_at="")

    plan = await service.probe_placement(cluster, _roles(decode=(1, 4)))

    assert plan["fits"] is True


@pytest.mark.asyncio
async def test_topology_is_unsupported_without_a_placement_profile(monkeypatch):
    monkeypatch.setattr(service, "_hardware_profiles", lambda: [get_profile("nvidia")])
    cluster = registry.Cluster(id="c1", name="c", description="", created_at="")

    with pytest.raises(ClusterOverviewError) as error:
        await service.get_topology(cluster)

    assert error.value.status_code == 409
    assert error.value.code == "topology_unsupported"


def test_a_claim_of_another_driver_with_the_same_pool_and_device_name_does_not_busy_a_gpu():
    nodes, slices = _cluster_nodes("n1")
    claim = _claim("n1", "n1-g0")
    claim["status"]["allocation"]["devices"]["results"][0]["driver"] = "dra.net"

    assert _topology(nodes, slices, [claim])["nodes"][0]["busyGpus"] == 0


@pytest.mark.asyncio
async def test_topology_is_unsupported_when_the_cluster_publishes_no_accelerators_of_a_placement_profile(monkeypatch):
    payloads = {"nodes": [_node("n1")], "resourceslices.resource.k8s.io": [], "resourceclaims.resource.k8s.io": []}

    async def fake_run_kubectl(argv, **_kwargs):
        return CommandResult(argv=tuple(argv), returncode=0, stdout=json.dumps({"items": payloads[argv[1]]}), stderr="")

    monkeypatch.setattr(service, "run_kubectl", fake_run_kubectl)
    cluster = registry.Cluster(id="c1", name="c", description="", created_at="")

    with pytest.raises(ClusterOverviewError) as error:
        await service.get_topology(cluster)

    assert error.value.code == "topology_unsupported"


@pytest.mark.asyncio
async def test_topology_and_placement_probe_are_limited_to_the_callers_clusters(monkeypatch):
    from types import SimpleNamespace

    from llm_d_lens.auth.contracts import Principal, RoleBinding, ScopeType
    from llm_d_lens.cluster.models import PlacementProbeRequest
    from llm_d_lens.core.exceptions import ForbiddenError

    router = importlib.import_module("llm_d_lens.cluster.router")

    binding = RoleBinding("end-user", ScopeType.CLUSTER, scope_cluster_id="mine")
    principal = Principal(user_id="u1", username="u1", bindings=(binding,), group_ids=frozenset())
    request = SimpleNamespace(state=SimpleNamespace(principal=principal))
    payload = PlacementProbeRequest.model_validate(
        {"roles": [{"name": "decode", "replicas": 1, "tensorParallelSize": 1}]}
    )
    reached = []
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: reached.append(cluster_id))

    with pytest.raises(ForbiddenError):
        await router.cluster_topology("other", request)
    with pytest.raises(ForbiddenError):
        await router.probe_placement("other", payload, request)
    assert reached == []
