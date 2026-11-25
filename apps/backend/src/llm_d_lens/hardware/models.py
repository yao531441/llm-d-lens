"""Typed hardware profile model.

The authoritative form of a profile is the JSON file under ``profiles/``
(validated against ``schema.json``); this module maps that data onto frozen
dataclasses for typed internal use, and provides the API response contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _str_map(value: Any) -> dict[str, str]:
    if not value:
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _options(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    return {str(key): item for key, item in value.items()}


def _matchers(value: Any) -> tuple[dict[str, str], ...]:
    if not value:
        return ()
    result: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append({str(key): str(entry) for key, entry in item.items()})
    return tuple(result)


@dataclass(frozen=True)
class AccessModeDriver:
    """One access mode's install configuration (device plugin or DRA driver)."""

    manifest_ref: str | None = None
    chart_version: str | None = None
    repo: str | None = None
    min_kubernetes_version: str | None = None
    daemonset_matchers: tuple[Mapping[str, str], ...] = ()
    monitoring_flag: str | None = None
    # Free-form, vendor-specific options. Each provider reads only the keys it
    # understands (e.g. NVIDIA plugin uses nfd_enabled / gfd_enabled /
    # device_plugin_all_nodes / runtime_class_name), keeping vendor specifics out
    # of the shared schema.
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AccessModeDriver:
        return cls(
            manifest_ref=data.get("manifest_ref"),
            chart_version=data.get("chart_version"),
            repo=data.get("repo"),
            min_kubernetes_version=data.get("min_kubernetes_version"),
            daemonset_matchers=_matchers(data.get("daemonset_matchers")),
            monitoring_flag=data.get("monitoring_flag"),
            options=_options(data.get("options")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_ref": self.manifest_ref,
            "chart_version": self.chart_version,
            "repo": self.repo,
            "min_kubernetes_version": self.min_kubernetes_version,
            "daemonset_matchers": [dict(item) for item in self.daemonset_matchers],
            "monitoring_flag": self.monitoring_flag,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class DriverContribution:
    """How to install the hardware's driver / device plugin.

    ``modes`` is keyed by access mode (``plugin``, ``dra``, or any vendor-defined
    mode), so a hardware that only supports one mode — or adds a new one — needs
    no schema change. ``nfd_manifest_refs`` is a shared prerequisite.
    """

    installer: str
    modes: Mapping[str, AccessModeDriver] = field(default_factory=dict)
    nfd_manifest_refs: tuple[str, ...] = ()

    def mode(self, access_mode: str) -> AccessModeDriver | None:
        return self.modes.get(access_mode)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriverContribution:
        return cls(
            installer=str(data.get("installer") or ""),
            modes={
                str(name): AccessModeDriver.from_dict(payload) for name, payload in (data.get("modes") or {}).items()
            },
            nfd_manifest_refs=_tuple_of_str(data.get("nfd_manifest_refs")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "installer": self.installer,
            "modes": {name: mode.to_dict() for name, mode in self.modes.items()},
            "nfd_manifest_refs": list(self.nfd_manifest_refs),
        }


@dataclass(frozen=True)
class HardwarePresence:
    """Blocking hardware-presence probe used by preflight."""

    node_label_selector: Mapping[str, str]
    ensure_refs: tuple[str, ...] = ()
    timeout_seconds: float = 90.0
    poll_interval_seconds: float = 5.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwarePresence:
        return cls(
            node_label_selector=_str_map(data.get("node_label_selector")),
            ensure_refs=_tuple_of_str(data.get("ensure_refs")),
            timeout_seconds=float(data.get("timeout_seconds", 90.0)),
            poll_interval_seconds=float(data.get("poll_interval_seconds", 5.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_label_selector": dict(self.node_label_selector),
            "ensure_refs": list(self.ensure_refs),
            "timeout_seconds": self.timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
        }


@dataclass(frozen=True)
class DeviceMetricSource:
    """A single scoped metric a consumer can selector-inject (profiling/evaluate).

    ``metric`` is the bare Prometheus series name, ``unit`` the canonical unit
    after ``scale`` is applied (e.g. a MiB gauge scales to bytes), and ``match``
    any extra label matchers the series needs (e.g.
    ``{"hw_gpu_task": "compute-all"}``). An empty ``metric`` (or an absent
    entry) means the hardware does not report that metric, so consumers must
    skip it entirely.
    """

    output_unit: str = ""
    aggregation: str = "avg"
    tile_aggregation: str = "avg"
    metric: str = ""
    unit: str = ""
    scale: float = 1.0
    match: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeviceMetricSource:
        try:
            scale = float(data.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        return cls(
            output_unit=str(data.get("output_unit") or data.get("unit") or ""),
            aggregation=str(data.get("aggregation") or "avg"),
            tile_aggregation=str(data.get("tile_aggregation") or "avg"),
            metric=str(data.get("metric") or ""),
            unit=str(data.get("unit") or ""),
            scale=scale,
            match=_str_map(data.get("match")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_unit": self.output_unit,
            "aggregation": self.aggregation,
            "tile_aggregation": self.tile_aggregation,
            "metric": self.metric,
            "unit": self.unit,
            "scale": self.scale,
            "match": dict(self.match),
        }


@dataclass(frozen=True)
class TelemetryContribution:
    """Metric sources and per-pod attribution for one hardware profile."""

    modes: Mapping[str, Any] = field(default_factory=dict)
    source_label: str = ""
    allocation: Mapping[str, Any] = field(default_factory=dict)
    provider_id: str | None = None
    device_metrics: Mapping[str, str] = field(default_factory=dict)
    # Selector-injectable, unit-aware view of the same metrics for consumers
    # that scope by namespace/pod or parse scraped text (profiling, evaluate).
    device_metric_sources: Mapping[str, DeviceMetricSource] = field(default_factory=dict)
    label_schema: Mapping[str, str] = field(default_factory=dict)
    allocation_join: str | None = None
    pod_metrics: Mapping[str, str] = field(default_factory=dict)
    scrape_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TelemetryContribution:
        return cls(
            modes=dict(data.get("modes") or {}),
            source_label=str(data.get("source_label") or ""),
            allocation=dict(data.get("allocation") or {}),
            provider_id=data.get("provider_id"),
            device_metrics=_str_map(data.get("device_metrics")),
            device_metric_sources={
                str(name): DeviceMetricSource.from_dict(payload)
                for name, payload in (data.get("device_metric_sources") or {}).items()
            },
            label_schema=_str_map(data.get("label_schema")),
            allocation_join=data.get("allocation_join"),
            pod_metrics=_str_map(data.get("pod_metrics")),
            scrape_path=data.get("scrape_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": dict(self.modes),
            "source_label": self.source_label,
            "allocation": dict(self.allocation),
            "provider_id": self.provider_id,
            "device_metrics": dict(self.device_metrics),
            "device_metric_sources": {name: source.to_dict() for name, source in self.device_metric_sources.items()},
            "label_schema": dict(self.label_schema),
            "allocation_join": self.allocation_join,
            "pod_metrics": dict(self.pod_metrics),
            "scrape_path": self.scrape_path,
        }


@dataclass(frozen=True)
class DeploymentContribution:
    """Overlay and resource-request shape for rendering deployments."""

    overlay_root: str
    arch: str
    modes: Mapping[str, Any] = field(default_factory=dict)
    managed_image_repositories: tuple[str, ...] = ()
    runtime_class_matchers: tuple[str, ...] = ()
    pci_attribute: str | None = None
    pci_attribute_domain: str | None = None
    device_class: str = ""
    claim_request_name: str = ""
    node_selector: Mapping[str, str] = field(default_factory=dict)
    resource_name: str | None = None
    supports_pci_allowlist: bool = False
    # Vendor's llm-d model-server image (e.g. llm-d-cuda vs llm-d-xpu); the
    # deploy default must follow the target hardware, not a hardcoded Intel image.
    runtime_image: str | None = None
    # Router topology some guides (e.g. tiered-prefix-cache) publish per-topology
    # values overlays for (single-host vs a multi-host LeaderWorkerSet, such as
    # multi-chip TPU pods); unset means "single-host", the topology every
    # currently registered profile uses.
    router_topology: str | None = None
    legacy_cli: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeploymentContribution:
        return cls(
            overlay_root=str(data.get("overlay_root") or ""),
            arch=str(data.get("arch") or ""),
            modes=dict(data.get("modes") or {}),
            managed_image_repositories=tuple(data.get("managed_image_repositories") or ()),
            runtime_class_matchers=tuple(data.get("runtime_class_matchers") or ()),
            pci_attribute=data.get("pci_attribute"),
            pci_attribute_domain=data.get("pci_attribute_domain"),
            device_class=str(data.get("device_class") or ""),
            claim_request_name=str(data.get("claim_request_name") or ""),
            node_selector=_str_map(data.get("node_selector")),
            resource_name=data.get("resource_name"),
            supports_pci_allowlist=bool(data.get("supports_pci_allowlist", False)),
            runtime_image=data.get("runtime_image"),
            router_topology=data.get("router_topology"),
            legacy_cli=_str_map(data.get("legacy_cli")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlay_root": self.overlay_root,
            "arch": self.arch,
            "modes": dict(self.modes),
            "managed_image_repositories": list(self.managed_image_repositories),
            "runtime_class_matchers": list(self.runtime_class_matchers),
            "pci_attribute": self.pci_attribute,
            "pci_attribute_domain": self.pci_attribute_domain,
            "device_class": self.device_class,
            "claim_request_name": self.claim_request_name,
            "node_selector": dict(self.node_selector),
            "resource_name": self.resource_name,
            "supports_pci_allowlist": self.supports_pci_allowlist,
            "runtime_image": self.runtime_image,
            "router_topology": self.router_topology,
            "legacy_cli": dict(self.legacy_cli),
        }


@dataclass(frozen=True)
class PlanningContribution:
    """Configuration/planning mapping for one hardware profile."""

    discovery: Mapping[str, Any] = field(default_factory=dict)
    memory_models: tuple[Mapping[str, Any], ...] = ()
    aic_system_patterns: tuple[str, ...] = ()
    aic_system_models: tuple[Mapping[str, str], ...] = ()
    model_label_keys: tuple[str, ...] = ()
    model_family_names: tuple[str, ...] = ()
    node_memory: Mapping[str, str] = field(default_factory=dict)
    default_backend: str = "vllm"
    device_class_allow_pattern: str = ""

    def aic_system_for_models(self, models: Iterable[str]) -> str | None:
        families = {name.lower() for name in self.model_family_names}
        systems = set()
        for model in models:
            if model.strip().lower() in families:
                continue
            matches = {
                row["system"]
                for row in self.aic_system_models
                if row.get("pattern") and re.search(row["pattern"], model, re.IGNORECASE)
            }
            if len(matches) != 1:
                return None
            systems.update(matches)
        return next(iter(systems)) if len(systems) == 1 else None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanningContribution:
        return cls(
            discovery=dict(data.get("discovery") or {}),
            memory_models=tuple(data.get("memory_models") or ()),
            aic_system_patterns=_tuple_of_str(data.get("aic_system_patterns")),
            aic_system_models=tuple(data.get("aic_system_models") or ()),
            model_label_keys=_tuple_of_str(data.get("model_label_keys")),
            model_family_names=_tuple_of_str(data.get("model_family_names")),
            node_memory=_str_map(data.get("node_memory")),
            default_backend=str(data.get("default_backend") or "vllm"),
            device_class_allow_pattern=str(data.get("device_class_allow_pattern") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery": dict(self.discovery),
            "memory_models": list(self.memory_models),
            "aic_system_patterns": list(self.aic_system_patterns),
            "aic_system_models": list(self.aic_system_models),
            "model_label_keys": list(self.model_label_keys),
            "model_family_names": list(self.model_family_names),
            "node_memory": dict(self.node_memory),
            "default_backend": self.default_backend,
            "device_class_allow_pattern": self.device_class_allow_pattern,
        }


@dataclass(frozen=True)
class PlacementContribution:
    """How DRA ResourceSlices expose PCIe-root topology for whole-root placement.

    A TP group is made of whole PCIe roots: every accelerator on the root plus, when ``nic_driver`` is set, one
    RDMA NIC on that root. Hardware without this contribution reports capacity totals only.
    """

    root_attribute: str
    nic_driver: str | None = None
    nic_rdma_attribute: str | None = None
    nic_numa_attribute: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlacementContribution:
        return cls(
            root_attribute=str(data["root_attribute"]),
            nic_driver=data.get("nic_driver"),
            nic_rdma_attribute=data.get("nic_rdma_attribute"),
            nic_numa_attribute=data.get("nic_numa_attribute"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_attribute": self.root_attribute,
            "nic_driver": self.nic_driver,
            "nic_rdma_attribute": self.nic_rdma_attribute,
            "nic_numa_attribute": self.nic_numa_attribute,
        }


@dataclass(frozen=True)
class UiContribution:
    """Presentation metadata for the browser client."""

    icon: str = "Cpu"
    labels: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UiContribution:
        return cls(icon=str(data.get("icon") or "Cpu"), labels=_str_map(data.get("labels")))

    def to_dict(self) -> dict[str, Any]:
        return {"icon": self.icon, "labels": dict(self.labels)}


@dataclass(frozen=True)
class HardwareProfile:
    """One accelerator family's declarative capabilities."""

    id: str
    display_name: str
    vendor: str
    request_model: str
    deployment: DeploymentContribution
    accelerator_keys: tuple[str, ...] = ()
    benchmark_profile: str = ""
    upstream_variant: str = ""
    upstream_vendor: str = ""
    device_classes: tuple[str, ...] = ()
    resource_prefixes: tuple[str, ...] = ()
    monitor_resource_suffixes: tuple[str, ...] = ()
    node_label_selector: Mapping[str, str] = field(default_factory=dict)
    dranet_device_class: str | None = None
    access_modes: tuple[str, ...] = ()
    driver: DriverContribution | None = None
    presence: HardwarePresence | None = None
    telemetry: TelemetryContribution | None = None
    evaluation_profile: str = ""
    planning: PlanningContribution = field(default_factory=PlanningContribution)
    placement: PlacementContribution | None = None
    ui: UiContribution = field(default_factory=UiContribution)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareProfile:
        driver = data.get("driver")
        presence = data.get("presence")
        telemetry = data.get("telemetry")
        placement = data.get("placement")
        return cls(
            id=str(data["id"]),
            display_name=str(data.get("display_name") or data["id"]),
            vendor=str(data.get("vendor") or ""),
            request_model=str(data.get("request_model") or "dra"),
            deployment=DeploymentContribution.from_dict(data.get("deployment") or {}),
            accelerator_keys=_tuple_of_str(data.get("accelerator_keys")),
            benchmark_profile=str(data.get("benchmark_profile") or ""),
            upstream_variant=str(data.get("upstream_variant") or ""),
            upstream_vendor=str(data.get("upstream_vendor") or ""),
            device_classes=_tuple_of_str(data.get("device_classes")),
            resource_prefixes=_tuple_of_str(data.get("resource_prefixes")),
            monitor_resource_suffixes=_tuple_of_str(data.get("monitor_resource_suffixes")),
            node_label_selector=_str_map(data.get("node_label_selector")),
            dranet_device_class=data.get("dranet_device_class"),
            access_modes=_tuple_of_str(data.get("access_modes")),
            driver=DriverContribution.from_dict(driver) if driver else None,
            presence=HardwarePresence.from_dict(presence) if presence else None,
            telemetry=TelemetryContribution.from_dict(telemetry) if telemetry else None,
            evaluation_profile=str(data.get("evaluation_profile") or data.get("benchmark_profile") or ""),
            planning=PlanningContribution.from_dict(data.get("planning") or {}),
            placement=PlacementContribution.from_dict(placement) if placement else None,
            ui=UiContribution.from_dict(data.get("ui") or {}),
        )

    @property
    def accelerator_aliases(self) -> frozenset[str]:
        """Every key that should resolve to this profile via accelerator key."""
        aliases = {self.id, self.benchmark_profile, self.upstream_variant}
        aliases.update(self.accelerator_keys)
        return frozenset(alias for alias in aliases if alias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "accelerator_keys": list(self.accelerator_keys),
            "benchmark_profile": self.benchmark_profile,
            "upstream_variant": self.upstream_variant,
            "upstream_vendor": self.upstream_vendor,
            "request_model": self.request_model,
            "device_classes": list(self.device_classes),
            "resource_prefixes": list(self.resource_prefixes),
            "monitor_resource_suffixes": list(self.monitor_resource_suffixes),
            "node_label_selector": dict(self.node_label_selector),
            "dranet_device_class": self.dranet_device_class,
            "access_modes": list(self.access_modes),
            "driver": self.driver.to_dict() if self.driver else None,
            "presence": self.presence.to_dict() if self.presence else None,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "deployment": self.deployment.to_dict(),
            "evaluation_profile": self.evaluation_profile,
            "planning": self.planning.to_dict(),
            "placement": self.placement.to_dict() if self.placement else None,
            "ui": self.ui.to_dict(),
        }


class HardwareCapabilitiesResponse(BaseModel):
    """``GET /api/v1/hardware/capabilities`` response contract."""

    version: str = Field(description="Content hash of all registered profiles.")
    profiles: list[dict[str, Any]] = Field(description="Registered hardware profiles.")
