"""Tests for AI candidate generation and deterministic validation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_d_lens.agentic.facts import PlanningHardwareSnapshot
from llm_d_lens.agentic.generator import (
    AgenticMcpClient,
    AICandidateGenerator,
    CandidateGenerationError,
    CandidateProposal,
    CandidateValidator,
)
from llm_d_lens.agentic.planner import (
    OpenAIPlannerSettings,
    PlanningFacts,
    WorkloadProfile,
    validate_request_context_budget,
)
from llm_d_lens.agentic.public_benchmark_record.case_retrieval import configuration_case_context
from llm_d_lens.ai_providers.client import AIProviderClientError, AIToolCall, AIToolTurn
from llm_d_lens.auth.security import verify_internal


def test_generator_schema_excludes_direct_vllm_and_accepts_optimized_baseline():
    proposal = {
        "provider_ref": "optimized-baseline",
        "replicas": 1,
        "tensor_parallel_size": 1,
        "rationale": "Use llm-d routing.",
    }
    assert CandidateProposal.model_validate(proposal).provider_ref == "optimized-baseline"
    assert "baseline-vllm" not in CandidateProposal.model_json_schema()["properties"]["provider_ref"]["enum"]
    with pytest.raises(ValidationError, match="provider_ref"):
        CandidateProposal.model_validate({**proposal, "provider_ref": "baseline-vllm"})


def test_generator_context_budget_counts_tools_and_prevents_external_completion(monkeypatch):
    class Tools:
        async def available_tools(self):
            cluster = _cluster_tool()
            cluster["description"] = "x" * 100000
            return [cluster, _aic_tool()]

        async def call(self, *_args, **_kwargs):
            pytest.fail("Over-budget request must not call MCP tools")

    class Provider:
        async def complete_tool_turn(self, **_kwargs):
            pytest.fail("Over-budget generator request must not reach the provider")

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner", context_window=65536),
        tools=Tools(),
    )
    with pytest.raises(CandidateGenerationError, match="Model context budget exceeded"):
        asyncio.run(generator.generate("cluster-a", "org/model", PlanningFacts(8, 32, 4), None))


def test_generator_reselects_complete_evidence_for_growing_actual_tool_history(monkeypatch):
    for module in ("generator", "planner", "facts"):
        monkeypatch.setattr(f"llm_d_lens.agentic.{module}.load_model_config", lambda _model: None)
    facts = PlanningFacts(
        8,
        32,
        4,
        configuration_cases=tuple(
            {
                "case_id": f"case:{index}",
                "configuration": {"source_path": "precise_prefix_cache_aware/results.json"},
                "measurements": [
                    {
                        "evidence_id": f"public-rag:{index}",
                        "metric": "ttft",
                        "unit": "ms",
                        "statistic": "p95",
                        "series": {"load": "x" * 20000},
                    }
                ],
            }
            for index in range(3)
        ),
        public_rag_evidence=tuple(
            {"evidence_id": f"public-rag:{index}", "value": index, "unit": "ms"} for index in range(3)
        ),
    )
    captured = []
    settings = OpenAIPlannerSettings(base_url="http://provider", model="planner")

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            if name == "get_cluster_overview":
                return {**_cluster_result(), "additional_history": "h" * 22000}
            return _aic_result()

    class Probe:
        async def complete_tool_turn(self, **kwargs):
            captured.append(json.loads(json.dumps(kwargs)))
            raise AIProviderClientError("probe")

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Probe())
    with pytest.raises(CandidateGenerationError, match="probe"):
        asyncio.run(AICandidateGenerator(settings, tools=Tools()).generate("cluster-a", "model", facts, None))
    full_request = {
        "model": settings.model,
        **{key: value for key, value in captured.pop().items() if key != "max_tokens"},
    }
    bound = 2 * len(json.dumps(full_request, ensure_ascii=True).encode("utf-8")) + 8192 + 1024
    settings = settings.model_copy(update={"context_window": bound - 1})

    class Provider:
        async def complete_tool_turn(self, **kwargs):
            request = {
                "model": settings.model,
                **{key: value for key, value in kwargs.items() if key != "max_tokens"},
            }
            validate_request_context_budget(settings, request, completion_tokens=8192)
            captured.append(json.loads(json.dumps(kwargs)))
            if len(captured) == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            if len(captured) == 2:
                return AIToolTurn(content="Continue gathering the final submission.")
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Historical reference; current SLO remains unknown.",
                                }
                            ]
                        },
                    ),
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(settings, tools=Tools())
    result = asyncio.run(generator.generate("cluster-a", "model", facts, None))
    assert result.resolved.facts.configuration_cases == facts.configuration_cases[:1]
    assert result.resolved.facts.public_rag_evidence == facts.public_rag_evidence[:1]
    assert result.resolved.facts.configuration_case_status["excluded"] == {
        "case:1": "request_context_budget",
        "case:2": "request_context_budget",
        "public-rag:1": "request_context_budget",
        "public-rag:2": "request_context_budget",
    }
    validated = CandidateValidator().validate(result.resolved.facts, result.proposals.candidates)
    assert validated.accepted[0].reference_material_ids == ["case:0", "public-rag:0"]
    assert "case:0" not in validated.accepted[0].evidence_ids
    for request in captured:
        serialized = json.dumps(request)
        assert "precise_prefix_cache_aware" not in serialized
        assert "configuration_case_evidence_ids" not in serialized
        assert "case:0" not in serialized and "public-rag:0" not in serialized
    contexts = [
        json.loads(request["messages"][0]["content"].split("Planning context (JSON):\n", 1)[1]) for request in captured
    ]
    assert (
        contexts[0]["configuration_cases"]
        == configuration_case_context(facts.configuration_cases[:2], {})["configuration_cases"]
    )
    assert (
        contexts[1]["configuration_cases"]
        == configuration_case_context(facts.configuration_cases[:1], {})["configuration_cases"]
    )
    assert (
        contexts[2]["configuration_cases"]
        == configuration_case_context(facts.configuration_cases[:1], {})["configuration_cases"]
    )
    assert captured[1]["messages"][1:] == captured[2]["messages"][1:3]
    assert len(captured[0]["messages"]) == 1
    assert facts.configuration_cases[2]["case_id"] == "case:2"


@pytest.mark.parametrize("operator_prompt", [None, "", "single GPU, no PD"])
def test_generator_keeps_prompt_and_nested_preference_consistent(monkeypatch, operator_prompt):
    captured = {}

    class Client:
        async def complete_tool_turn(self, **kwargs):
            captured.update(kwargs)
            raise AIProviderClientError("stop after capturing input")

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Client())
    facts = PlanningFacts(8, 32, 4, operator_preference="prefer existing cache")
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="generator"), tools=Tools()
    )
    with pytest.raises(CandidateGenerationError, match="stop after capturing input"):
        asyncio.run(generator.generate("cluster-a", "model-a", facts, operator_prompt))
    preference = facts.operator_preference if operator_prompt is None else operator_prompt
    content = captured["messages"][0]["content"]
    assert content.startswith("Operator preference")
    prefix, context = content.split("\n\nPlanning context (JSON):\n", 1)
    assert prefix.endswith(preference or "(not provided)")
    payload = json.loads(context)
    assert payload["operator_preference"] == preference
    assert payload["deployment_intent"]["operator_preference"] == preference
    assert facts.operator_preference == "prefer existing cache"


def test_validator_keeps_valid_ai_candidates_when_other_proposals_are_invalid():
    facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        gpu_memory_utilization=0.6,
        evidence_ids=("aic:search",),
    )
    proposals = [
        CandidateProposal(
            provider_ref="optimized-baseline",
            replicas=1,
            tensor_parallel_size=1,
            rationale="Small deployable baseline.",
            evidence_ids=["aic:search"],
        ),
        CandidateProposal(
            provider_ref="pd-disaggregation",
            replicas=2,
            tensor_parallel_size=2,
            prefill_replicas=2,
            prefill_tensor_parallel_size=2,
            rationale="High-throughput topology.",
        ),
    ]

    result = CandidateValidator().validate(facts, proposals)

    assert [candidate.id for candidate in result.accepted] == ["optimized-baseline-tp1-r1"]
    assert result.accepted[0].gpu_memory_utilization == 0.6
    assert result.accepted[0].evidence_ids == ["aic:search"]
    assert [candidate.id for candidate in result.rejected] == ["pd-disaggregation-p2-tp2-d2-tp2"]
    assert result.rejected[0].rejection_reasons == ["insufficient free accelerator cards"]


def test_mcp_client_signs_authenticated_principal(monkeypatch):
    monkeypatch.setenv("LENS_INTERNAL_AUTH_SECRET", "test-secret")
    monkeypatch.setattr("llm_d_lens.agentic.generator.time.time", lambda: 1000)

    headers = AgenticMcpClient("https://127.0.0.1:3005/api/assistant/mcp", principal_id="user-1")._headers()

    assert headers["x-prism-principal-id"] == "user-1"
    assert verify_internal(
        "test-secret",
        signature=headers["x-prism-internal-sig"],
        timestamp=1000,
        method="POST",
        path="/api/assistant/mcp",
        principal_id="user-1",
        now=1000,
    )


def test_validator_drops_ai_evidence_ids_not_present_in_authoritative_facts():
    facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        evidence_ids=("cluster-overview:cluster-a",),
    )
    proposal = CandidateProposal(
        provider_ref="optimized-baseline",
        replicas=1,
        tensor_parallel_size=1,
        rationale="Fits.",
        evidence_ids=["cluster-overview:cluster-a", "invented:evidence"],
    )

    result = CandidateValidator().validate(facts, [proposal])

    assert result.accepted[0].evidence_ids == ["cluster-overview:cluster-a"]


def test_validator_records_supplied_reference_material_without_claiming_model_usage():
    facts = PlanningFacts(
        8,
        32,
        4,
        configuration_cases=(
            {
                "case_id": "case:authorized",
                "measurements": [{"evidence_id": "benchmark:authorized"}],
            },
        ),
    )
    proposal = CandidateProposal(
        provider_ref="optimized-baseline",
        replicas=1,
        tensor_parallel_size=1,
        rationale="Adapt the supplied case.",
        evidence_ids=["case:authorized", "benchmark:authorized", "case:invented"],
    )
    result = CandidateValidator().validate(facts, [proposal])
    assert result.accepted[0].reference_material_ids == ["case:authorized", "benchmark:authorized"]
    assert not result.accepted[0].evidence_ids


def test_generator_schema_rejects_unknown_fields():
    field = "unknown_field"
    assert field not in CandidateProposal.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CandidateProposal(
            provider_ref="optimized-baseline",
            replicas=1,
            tensor_parallel_size=1,
            rationale="Bounded proposal.",
            **{field: ["invented"]},
        )


def test_candidate_proposal_normalizes_pd_decode_aliases():
    proposal = CandidateProposal.model_validate(
        {
            "provider_ref": "pd-disaggregation",
            "decode_replicas": 2,
            "decode_tensor_parallel_size": 1,
            "prefill_replicas": 1,
            "prefill_tensor_parallel_size": 1,
            "rationale": "Distributed topology.",
        }
    )

    assert proposal.replicas == 2
    assert proposal.tensor_parallel_size == 1


def test_validator_rejects_incomplete_pd_topology():
    facts = PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=32, free_gpu_count=8)
    proposal = CandidateProposal(
        provider_ref="pd-disaggregation",
        replicas=1,
        tensor_parallel_size=1,
        rationale="Incomplete proposal.",
    )

    result = CandidateValidator().validate(facts, [proposal])

    assert result.accepted == []
    assert result.rejected[0].rejection_reasons == [
        "PD candidates require both prefill replicas and tensor parallelism",
    ]


def _cluster_tool():
    return {
        "name": "get_cluster_overview",
        "description": "[read] Get cluster.",
        "input_schema": {"type": "object", "properties": {"clusterId": {"type": "string"}}},
    }


def _cluster_result():
    return {
        "hardware": {
            "gpuCount": 4,
            "availableGpuCount": 4,
            "vramBytes": 4 * 32 * 1024**3,
            "memoryBytes": 128 * 1024**3,
            "memoryUsagePercent": 25,
        }
    }


def _aic_tool():
    return {
        "name": "search_candidates",
        "description": "[read] Search AIC candidates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "workload": {"type": "object"},
                "searchConfig": {"type": "object"},
                "sourceIds": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["workload", "searchConfig", "sourceIds"],
        },
    }


def _aic_arguments():
    return {
        "workload": {"model": "Qwen/Qwen3-8B", "isl": 4000, "osl": 1000},
        "searchConfig": {"totalGpus": 4},
        "sourceIds": ["aic"],
    }


def _aic_result(*, ttft_ms=300, tpot_ms=30):
    return {
        "candidates": [
            {
                "source": "aic",
                "topologyMode": "agg",
                "decodeTp": 1,
                "decodeReplicas": 1,
                "predicted": {"ttftMs": ttft_ms, "tpotMs": tpot_ms, "throughputTps": 1000},
            }
        ]
    }


def _mcp_cluster_result():
    return {
        "cluster": {"id": "cluster-a"},
        "kubernetes": {
            "hardware": {
                "gpuCount": 8,
                "availableGpuCount": 6,
                "vramBytes": 8 * 24 * 1024**3,
                "memoryBytes": 256 * 1024**3,
                "memoryUsagePercent": 25,
            },
            "nodes": [],
        },
    }


@pytest.mark.parametrize("nested", [False, True])
def test_resolve_mcp_facts_rejects_missing_live_vram_despite_retired_overrides(monkeypatch, nested):
    monkeypatch.setenv("AGENTIC_TEST_CLUSTER_VRAM_GIB_OVERRIDES", '{"cluster-a":80}')
    overview = _mcp_cluster_result()
    overview["kubernetes"]["hardware"]["vramBytes"] = 0
    with pytest.raises(CandidateGenerationError, match="no usable accelerator VRAM"):
        AICandidateGenerator._resolve_mcp_facts(
            "cluster-a",
            PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=1, free_gpu_count=0),
            {"get_cluster_overview": [overview if nested else overview["kubernetes"]]},
        )
    assert overview["kubernetes"]["hardware"]["vramBytes"] == 0


def test_resolve_mcp_facts_accepts_nested_cluster_overview_hardware():
    resolved = AICandidateGenerator._resolve_mcp_facts(
        "cluster-a",
        PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=1, free_gpu_count=0),
        {"get_cluster_overview": [_mcp_cluster_result()]},
    )

    assert resolved.facts.vram_per_gpu_gib == 24
    assert resolved.facts.free_gpu_count == 6
    assert resolved.facts.cpu_buffer_gib == 192
    assert resolved.cluster_snapshot["cluster"]["id"] == "cluster-a"


@pytest.mark.parametrize(
    "accelerators,expected_system",
    [
        (None, None),
        ([{"id": "nvidia", "gpuCount": 4, "models": ["NVIDIA-A100-80GB-PCIe", "ampere"]}], "a100_pcie"),
        ([{"id": "unknown", "models": ["NVIDIA-A100-80GB-PCIe"]}], None),
        ([{"id": "nvidia", "gpuCount": 4, "models": ["A100 PCIe", "unknown"]}], None),
        ([{"id": "nvidia", "gpuCount": 4, "models": ["A100 PCIe", "A100 SXM"]}], None),
        ([{"id": "nvidia", "gpuCount": 4, "models": ["H100"]}], "h100_sxm"),
    ],
)
@pytest.mark.parametrize("profile", [None, WorkloadProfile(), WorkloadProfile(shared_prefix_ratio=0)])
@pytest.mark.parametrize("shared_snapshot", [False, True])
def test_ai_generator_gets_required_evidence_via_model_selected_mcp_call(
    monkeypatch, accelerators, expected_system, profile, shared_snapshot
):
    monkeypatch.setenv("AGENTIC_TEST_CLUSTER_HARDWARE_OVERRIDES", '{"cluster-a":"H200"}')
    monkeypatch.setenv("AGENTIC_TEST_CLUSTER_VRAM_GIB_OVERRIDES", '{"cluster-a":96}')
    captured = {"tool_args": []}
    discoveries = []

    async def overview(cluster):
        discoveries.append(cluster.id)
        result = _cluster_result()
        if accelerators is not None:
            result["hardware"]["accelerators"] = accelerators
        return result

    monkeypatch.setattr(
        "llm_d_lens.agentic.facts.registry.require_cluster", lambda cluster_id: SimpleNamespace(id=cluster_id)
    )
    monkeypatch.setattr("llm_d_lens.agentic.facts.build_overview", overview)

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, arguments):
            captured["tool_args"].append((name, arguments))
            if name == "get_cluster_overview":
                cluster = _cluster_result()
                if accelerators is not None:
                    cluster["hardware"]["accelerators"] = accelerators
                return cluster
            return _aic_result()

    class Provider:
        calls = 0

        async def complete_tool_turn(self, **kwargs):
            captured.update(kwargs)
            self.calls += 1
            if self.calls == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="call-1", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(
                            id="call-2",
                            name="search_candidates",
                            arguments={
                                **_aic_arguments(),
                                "searchConfig": {"aicSystemName": "a100", "totalGpus": 8},
                            },
                        ),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit-1",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Fits the current cluster.",
                                    "evidence_ids": ["cluster-overview:cluster-a"],
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )
    facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        evidence_ids=("aic:search",),
        public_rag_evidence=({"evidence_id": "public-rag:published-1", "citation": {"source_revision": "rev-1"}},),
        workload_profile=profile,
    )

    async def generate():
        snapshot = PlanningHardwareSnapshot("cluster-a") if shared_snapshot else None
        if snapshot is not None:
            first = await snapshot.get("cluster-a")
            first["hardware"]["availableGpuCount"] = 0
        return await generator.generate(
            "cluster-a", "Qwen/Qwen3-8B", facts, "prefer throughput", hardware_snapshot=snapshot
        )

    proposals = asyncio.run(generate())

    assert discoveries == (["cluster-a"] if shared_snapshot else [])
    assert captured["tool_args"] == [
        *([] if shared_snapshot else [("get_cluster_overview", {"clusterId": "cluster-a"})]),
        (
            "search_candidates",
            {
                **_aic_arguments(),
                "searchConfig": {"totalGpus": 4, **({"aicSystemName": expected_system} if expected_system else {})},
            },
        ),
    ]
    assert "The server will independently recompute" in captured["system_prompt"]
    assert "Do not score, select, render, approve, or deploy" in captured["system_prompt"]
    user_content = captured["messages"][0]["content"]
    assert user_content.startswith("Operator preference")
    assert user_content.index("prefer throughput") < user_content.index("Planning context (JSON):")
    user_context = json.loads(user_content.split("Planning context (JSON):\n", 1)[1])
    assert next(iter(user_context)) == "operator_preference"
    assert user_context["operator_preference"] == "prefer throughput"
    assert "evidence_id" not in user_context["published_benchmark_evidence"][0]
    assert "citation" not in user_context["published_benchmark_evidence"][0]
    assert "public_rag_evidence" not in user_context
    assert "RAG" not in captured["system_prompt"]
    assert "structured SQL database" in captured["system_prompt"]
    assert "missing evidence for this request" in captured["system_prompt"]
    assert user_context["explicit_workload"] == (
        {"shared_prefix_ratio": 0} if profile and profile.shared_prefix_ratio == 0 else {}
    )
    assert "takes precedence over AIC estimates" in captured["system_prompt"]
    assert proposals.resolved.cluster_snapshot["hardware"].get("accelerators") == (
        accelerators if accelerators is not None else _cluster_result()["hardware"].get("accelerators")
    )
    assert proposals.resolved.facts.public_rag_evidence == facts.public_rag_evidence
    assert "validated_planning_facts" not in user_context
    assert proposals.proposals.candidates[0].provider_ref == "optimized-baseline"
    assert proposals.resolved.facts.vram_per_gpu_gib == 32
    assert proposals.resolved.facts.free_gpu_count == 4


@pytest.mark.parametrize("discovered", [False, True])
def test_ai_generator_relaxes_only_after_empty_aic_search_and_freezes_original_targets(monkeypatch, discovered):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool(), {**_aic_tool(), "name": "search_aic_with_relaxation"}]

        async def call(self, name, arguments):
            calls.append((name, arguments))
            if name == "get_cluster_overview":
                cluster = _cluster_result()
                if discovered:
                    cluster["hardware"]["accelerators"] = [
                        {"id": "nvidia", "gpuCount": 4, "models": ["NVIDIA-A100-80GB-PCIe", "ampere"]}
                    ]
                return cluster
            if name == "search_candidates":
                return {"candidates": []}
            anchor = {**_aic_result(ttft_ms=600)["candidates"][0], "decodeTp": 2, "totalGpus": 2}
            return {
                "candidates": [anchor],
                "anchor": anchor,
                "sloSatisfied": False,
            }

    class Provider:
        rounds = 0

        async def complete_tool_turn(self, **kwargs):
            self.rounds += 1
            if self.rounds == 1:
                assert "search_aic_with_relaxation" not in [
                    AICandidateGenerator._provider_tool_name(tool) for tool in kwargs["tools"]
                ]
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            if self.rounds == 2:
                assert "search_aic_with_relaxation" in [
                    AICandidateGenerator._provider_tool_name(tool) for tool in kwargs["tools"]
                ]
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id="aic",
                            name="search_aic_with_relaxation",
                            arguments={
                                "clusterId": "wrong",
                                "workload": {"ttftMs": 1000},
                                "searchConfig": {"totalGpus": 64},
                            },
                        ),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 2,
                                    "tensor_parallel_size": 1,
                                    "rationale": "AIC anchor requires validation.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    result = asyncio.run(
        AICandidateGenerator(
            OpenAIPlannerSettings(base_url="http://provider", model="planner"),
            tools=Tools(),
        ).generate(
            "cluster-a",
            "model",
            PlanningFacts(8, 32, 4, ttft_slo_ms=500),
            None,
        )
    )

    assert calls[1] == (
        "search_candidates",
        _aic_arguments()
        | {
            "searchConfig": {"totalGpus": 4, **({"aicSystemName": "a100_pcie"} if discovered else {})},
            "workload": {
                "model": "model",
                "isl": 4000,
                "osl": 1000,
                "ttftMs": 500,
            },
        },
    )
    assert calls[2] == (
        "search_aic_with_relaxation",
        {
            "clusterId": "cluster-a",
            "workload": {"model": "model", "isl": 4000, "osl": 1000, "ttftMs": 500},
            "searchConfig": {
                "capacityMode": "free",
                "customGpuCount": None,
                **({"aicSystemName": "a100_pcie"} if discovered else {}),
            },
        },
    )
    assert result.resolved.facts.aic_predictions == ()
    assert result.proposals.candidates[0].replicas == 2
    assert [
        (candidate.provider_ref, candidate.tensor_parallel_size, candidate.replicas)
        for candidate in result.proposals.candidates
    ] == [
        ("optimized-baseline", 1, 2),
        ("optimized-baseline", 2, 1),
    ]
    accepted = CandidateValidator().validate(result.resolved.facts, result.proposals.candidates).accepted
    assert accepted[-1].required_gpus == 2


@pytest.mark.parametrize(
    "capacity_mode,custom_gpu_count,expected_budget",
    [
        ("free", None, 3),
        ("all", None, 8),
        ("custom", 2, 2),
    ],
)
def test_ai_generator_relaxes_no_feasible_aic_error_with_live_gpu_budget(
    monkeypatch,
    capacity_mode,
    custom_gpu_count,
    expected_budget,
):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool(), {**_aic_tool(), "name": "search_aic_with_relaxation"}]

        async def call(self, name, arguments):
            calls.append((name, arguments))
            if name == "get_cluster_overview":
                cluster = _mcp_cluster_result()
                cluster["kubernetes"]["hardware"]["availableGpuCount"] = 3
                return cluster
            if name == "search_candidates":
                return {
                    "error": "upstream request failed (status 400)",
                    "detail": {"error": "No feasible configurations found for the given parameters."},
                }
            return {"candidates": [], "anchor": None}

    class Provider:
        rounds = 0

        async def complete_tool_turn(self, **kwargs):
            self.rounds += 1
            if self.rounds == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(
                            id="aic",
                            name="search_candidates",
                            arguments={
                                **_aic_arguments(),
                                "searchConfig": {"totalGpus": 8},
                            },
                        ),
                    ]
                )
            if self.rounds == 2:
                assert "search_aic_with_relaxation" in [
                    AICandidateGenerator._provider_tool_name(tool) for tool in kwargs["tools"]
                ]
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id="relaxed",
                            name="search_aic_with_relaxation",
                            arguments={
                                "clusterId": "cluster-a",
                                "workload": {},
                                "searchConfig": {},
                            },
                        )
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "No AIC prediction is available.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    result = asyncio.run(
        AICandidateGenerator(
            OpenAIPlannerSettings(base_url="http://provider", model="planner"),
            tools=Tools(),
        ).generate(
            "cluster-a",
            "Qwen/Qwen3-8B",
            PlanningFacts(
                8,
                24,
                0,
                hardware_capacity_mode=capacity_mode,
                custom_gpu_count=custom_gpu_count,
            ),
            None,
        )
    )

    assert calls[1][0] == "search_candidates"
    assert calls[1][1]["searchConfig"]["totalGpus"] == expected_budget
    assert calls[2][0] == "search_aic_with_relaxation"
    assert result.proposals.tool_trace[1]["result"]["candidates"] == []
    assert (
        result.proposals.tool_trace[1]["result"]["detail"]["error"]
        == "No feasible configurations found for the given parameters."
    )
    assert result.resolved.facts.free_gpu_count == expected_budget


def test_ai_generator_keeps_relaxed_tool_hidden_when_aic_returns_candidates(monkeypatch):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool(), {**_aic_tool(), "name": "search_aic_with_relaxation"}]

        async def call(self, name, arguments):
            calls.append(name)
            return _cluster_result() if name == "get_cluster_overview" else _aic_result(ttft_ms=600)

    class Provider:
        rounds = 0

        async def complete_tool_turn(self, **kwargs):
            self.rounds += 1
            assert "search_aic_with_relaxation" not in [
                AICandidateGenerator._provider_tool_name(tool) for tool in kwargs["tools"]
            ]
            if self.rounds == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Initial AIC search returned a candidate.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    asyncio.run(
        AICandidateGenerator(
            OpenAIPlannerSettings(base_url="http://provider", model="planner"),
            tools=Tools(),
        ).generate("cluster-a", "model", PlanningFacts(8, 32, 4, ttft_slo_ms=500), None)
    )

    assert calls == ["get_cluster_overview", "search_candidates"]


def test_ai_generator_does_not_relax_unrelated_aic_error(monkeypatch):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool(), {**_aic_tool(), "name": "search_aic_with_relaxation"}]

        async def call(self, name, _arguments):
            calls.append(name)
            if name == "get_cluster_overview":
                return _cluster_result()
            return {"error": "upstream request failed (status 400)", "detail": {"error": "Model not supported"}}

    class Provider:
        rounds = 0

        async def complete_tool_turn(self, **kwargs):
            self.rounds += 1
            if self.rounds == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            assert "search_aic_with_relaxation" not in [
                AICandidateGenerator._provider_tool_name(tool) for tool in kwargs["tools"]
            ]
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "No prediction available.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    result = asyncio.run(
        AICandidateGenerator(
            OpenAIPlannerSettings(base_url="http://provider", model="planner"),
            tools=Tools(),
        ).generate("cluster-a", "model", PlanningFacts(8, 32, 4), None)
    )

    assert calls == ["get_cluster_overview", "search_candidates"]
    assert "candidates" not in result.proposals.tool_trace[1]["result"]


def test_ai_generator_does_not_search_when_no_gpus_are_available(monkeypatch):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            calls.append(name)
            cluster = _cluster_result()
            cluster["hardware"]["availableGpuCount"] = 0
            return cluster

    class Provider:
        async def complete_tool_turn(self, **_kwargs):
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(id="cluster", name="get_cluster_overview", arguments={}),
                    AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    with pytest.raises(CandidateGenerationError, match="No GPUs available"):
        asyncio.run(
            AICandidateGenerator(
                OpenAIPlannerSettings(base_url="http://provider", model="planner"),
                tools=Tools(),
            ).generate("cluster-a", "model", PlanningFacts(8, 32, 0), None)
        )
    assert calls == ["get_cluster_overview"]


def test_ai_generator_requires_relaxed_search_after_empty_aic_result(monkeypatch):
    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool(), {**_aic_tool(), "name": "search_aic_with_relaxation"}]

        async def call(self, name, _arguments):
            return _cluster_result() if name == "get_cluster_overview" else {"candidates": []}

    class Provider:
        rounds = 0

        async def complete_tool_turn(self, **_kwargs):
            self.rounds += 1
            if self.rounds == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "No AIC result.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    with pytest.raises(CandidateGenerationError, match="skipped relaxed AIC search"):
        asyncio.run(
            AICandidateGenerator(
                OpenAIPlannerSettings(base_url="http://provider", model="planner"),
                tools=Tools(),
            ).generate("cluster-a", "model", PlanningFacts(8, 32, 4), None)
        )


def test_ai_generator_does_not_replay_an_empty_assistant_response(monkeypatch):
    request_messages = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            return _cluster_result() if name == "get_cluster_overview" else _aic_result()

    class Provider:
        calls = 0

        async def complete_tool_turn(self, **kwargs):
            self.calls += 1
            request_messages.append(json.loads(json.dumps(kwargs["messages"])))
            if self.calls == 1:
                return AIToolTurn()
            if self.calls == 2:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Fits the observed cluster.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    result = asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))

    assert result.proposals.candidates[0].provider_ref == "optimized-baseline"
    assert request_messages[1] == [
        request_messages[0][0],
        {
            "role": "user",
            "content": (
                "Your response did not call a tool. Continue by calling one provided tool; do not reply with text only."
            ),
        },
    ]


@pytest.mark.parametrize(
    "optional_names",
    [
        ("list_simulation_tasks",),
        ("list_simulation_tasks", "list_deployment_executions", "list_evaluate_runs", "list_model_cache_entries"),
    ],
)
def test_ai_generator_runs_bounded_optional_mcp_calls_before_final_candidates(monkeypatch, optional_names):
    calls = []
    exposed_tools = []

    class Tools:
        async def available_tools(self):
            return [
                _cluster_tool(),
                _aic_tool(),
                *(
                    {
                        "name": name,
                        "description": "[read] List planning history.",
                        "input_schema": {"type": "object", "properties": {"model": {"type": "string"}}},
                    }
                    for name in optional_names
                ),
            ]

        async def call(self, name, arguments):
            calls.append((name, arguments))
            if name == "get_cluster_overview":
                return _cluster_result()
            if name == "search_candidates":
                return {"candidates": []}
            return {"items": [{"id": "simulation-1", "status": "completed"}]}

    class Provider:
        turns = 0

        async def complete_tool_turn(self, **kwargs):
            self.turns += 1
            exposed_tools.append([AICandidateGenerator._provider_tool_name(tool) for tool in kwargs["tools"]])
            if self.turns == 1:
                assert (
                    "at most 4 MCP tool calls in one response and 8 across at most 4 tool rounds"
                    in kwargs["system_prompt"]
                )
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="call-1", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="call-2", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            if self.turns == 2:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id=f"call-{index}",
                            name=name,
                            arguments={"model": "Qwen/Qwen3-8B"},
                        )
                        for index, name in enumerate(optional_names, start=3)
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit-1",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 2,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Uses simulation history.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    proposals = asyncio.run(
        generator.generate(
            "cluster-a",
            "Qwen/Qwen3-8B",
            PlanningFacts(8, 32, 4),
            "prefer throughput",
        )
    )

    assert calls == [
        ("get_cluster_overview", {"clusterId": "cluster-a"}),
        ("search_candidates", _aic_arguments()),
        *((name, {"model": "Qwen/Qwen3-8B"}) for name in optional_names),
    ]
    assert set(exposed_tools[0]) == {"get_cluster_overview", "search_candidates"}
    assert set(optional_names) <= set(exposed_tools[1])
    assert "submit_candidate_proposals" in exposed_tools[1]
    assert proposals.proposals.candidates[0].provider_ref == "optimized-baseline"
    assert proposals.resolved.facts.aic_predictions == ()
    assert "aic:search" not in proposals.resolved.facts.evidence_ids
    assert [trace["tool"] for trace in proposals.proposals.tool_trace[2:]] == list(optional_names)
    assert all(trace["status"] == "success" for trace in proposals.proposals.tool_trace[2:])
    assert len(proposals.proposals.tool_trace[2]["arguments_digest"]) == 64
    assert len(proposals.proposals.tool_trace[2]["result_digest"]) == 64
    assert proposals.proposals.tool_trace[2]["arguments"] == {"model": "Qwen/Qwen3-8B"}
    assert proposals.proposals.tool_trace[2]["result"] == {"items": [{"id": "simulation-1", "status": "completed"}]}
    aic_trace = proposals.proposals.tool_trace[1]
    assert aic_trace["arguments"] == _aic_arguments()
    assert aic_trace["result"] == {"candidates": []}


def test_ai_generator_retries_one_text_only_turn(monkeypatch):
    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            return _cluster_result() if name == "get_cluster_overview" else {"candidates": []}

    class Provider:
        turns = 0

        async def complete_tool_turn(self, **kwargs):
            self.turns += 1
            if self.turns == 1:
                return AIToolTurn(content="I will inspect the cluster first.")
            if self.turns == 2:
                assert "did not call a tool" in kwargs["messages"][-1]["content"]
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="call-1", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="call-2", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit-1",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Fits.",
                                }
                            ]
                        },
                    )
                ]
            )

    provider = Provider()
    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: provider)
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    result = asyncio.run(
        generator.generate(
            "cluster-a",
            "Qwen/Qwen3-8B",
            PlanningFacts(
                8,
                32,
                4,
                ttft_slo_ms=40,
                workload_profile={"mean_input_tokens": 4096, "mean_output_tokens": 512},
            ),
            None,
        )
    )

    assert provider.turns == 3
    assert result.proposals.candidates[0].provider_ref == "optimized-baseline"


def test_ai_generator_rejects_tool_outside_planning_allowlist(monkeypatch):
    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, *_args):
            raise AssertionError("blocked tools must not reach MCP")

    class Provider:
        async def complete_tool_turn(self, **_kwargs):
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="call-1",
                        name="approve_agentic_plan",
                        arguments={"run_id": "run-1"},
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    with pytest.raises(CandidateGenerationError, match="not allowed"):
        asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))


def test_ai_generator_rejects_submission_without_cluster_evidence(monkeypatch):
    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, *_args):
            raise AssertionError("submission must not execute MCP tools")

    class Provider:
        async def complete_tool_turn(self, **_kwargs):
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit-1",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Unsupported guess.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    with pytest.raises(CandidateGenerationError, match="invalid response: cluster evidence was not gathered"):
        asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))


def test_ai_generator_enforces_per_round_mcp_call_limit(monkeypatch):
    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, *_args):
            raise AssertionError("oversized batch must be rejected before execution")

    class Provider:
        calls = 0

        async def complete_tool_turn(self, **_kwargs):
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(id=f"call-{index}", name="get_cluster_overview", arguments={"clusterId": "cluster-a"})
                    for index in range(5)
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    with pytest.raises(CandidateGenerationError, match=r"too many tool calls in round 1: 5 \(limit 4\)"):
        asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))


def test_ai_generator_allows_only_four_mcp_rounds(monkeypatch):
    class Tools:
        calls = 0

        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            self.calls += 1
            return _cluster_result() if name == "get_cluster_overview" else {"candidates": []}

    class Provider:
        calls = 0

        async def complete_tool_turn(self, **_kwargs):
            if self.calls == 0:
                self.calls += 1
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="repeat",
                        name="get_cluster_overview",
                        arguments={"clusterId": "cluster-a"},
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    tools = Tools()
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=tools,
    )

    with pytest.raises(CandidateGenerationError, match="tool-round limit"):
        asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))

    assert tools.calls == 5


def test_ai_generator_rejects_first_tool_batch_without_aic_search(monkeypatch):
    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, *_args):
            raise AssertionError("invalid first batch must not execute MCP tools")

    class Provider:
        turns = 0

        async def complete_tool_turn(self, **_kwargs):
            self.turns += 1
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="cluster",
                        name="get_cluster_overview",
                        arguments={"clusterId": "cluster-a"},
                    )
                ]
            )

    provider = Provider()
    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: provider)
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    with pytest.raises(CandidateGenerationError, match="must first call"):
        asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))

    assert provider.turns == 2


def test_ai_generator_retries_invalid_first_tool_batch_once(monkeypatch):
    calls = []
    requests = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            calls.append(name)
            return _cluster_result() if name == "get_cluster_overview" else _aic_result()

    class Provider:
        async def complete_tool_turn(self, **kwargs):
            requests.append(list(kwargs["messages"]))
            if len(requests) == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id="cluster-only",
                            name="get_cluster_overview",
                            arguments={"clusterId": "cluster-a"},
                        )
                    ]
                )
            if len(requests) == 2:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Fits the observed cluster.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    result = asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))

    assert len(requests) == 3
    assert "Retry once by calling exactly" in requests[1][-1]["content"]
    assert calls == ["get_cluster_overview", "search_candidates"]
    assert result.proposals.candidates[0].provider_ref == "optimized-baseline"


def test_ai_generator_injects_required_cluster_and_aic_arguments(monkeypatch):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, arguments):
            calls.append((name, arguments))
            return _cluster_result() if name == "get_cluster_overview" else _aic_result()

    class Provider:
        turn = 0

        async def complete_tool_turn(self, **_kwargs):
            self.turn += 1
            if self.turn > 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id="submit",
                            name="submit_candidate_proposals",
                            arguments={
                                "candidates": [
                                    {
                                        "provider_ref": "optimized-baseline",
                                        "replicas": 1,
                                        "tensor_parallel_size": 1,
                                        "rationale": "Fits the observed cluster.",
                                    }
                                ]
                            },
                        )
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "wrong-cluster"}),
                    AIToolCall(
                        id="aic",
                        name="search_candidates",
                        arguments={
                            **_aic_arguments(),
                            "sourceIds": ["manual"],
                            "workload": {"ttftMs": 35, "tpotMs": 20},
                        },
                    ),
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    result = asyncio.run(
        generator.generate(
            "cluster-a",
            "Qwen/Qwen3-8B",
            PlanningFacts(
                8,
                32,
                4,
                ttft_slo_ms=40,
                workload_profile={"mean_input_tokens": 4096, "mean_output_tokens": 512},
            ),
            None,
        )
    )

    assert result.proposals.candidates[0].provider_ref == "optimized-baseline"
    assert calls[0] == ("get_cluster_overview", {"clusterId": "cluster-a"})
    assert calls[1][0] == "search_candidates"
    assert calls[1][1]["sourceIds"] == ["aic"]
    assert calls[1][1]["workload"] == {
        "model": "Qwen/Qwen3-8B",
        "isl": 4096,
        "osl": 512,
        "ttftMs": 40,
    }
    assert result.proposals.tool_trace[1]["arguments"]["workload"] == calls[1][1]["workload"]


def test_aic_tool_arguments_use_requested_workload_and_slo():
    calls = [
        AIToolCall(
            id="aic",
            name="search_candidates",
            arguments={
                "workload": {"isl": 8, "osl": 8, "ttftMs": 999, "tpotMs": 999},
            },
        )
    ]
    normalized = AICandidateGenerator._normalize_required_tool_arguments(
        calls,
        "cluster-a",
        "Qwen/Qwen3-0.6B",
        PlanningFacts(
            0.7,
            32,
            4,
            ttft_slo_ms=500,
            tpot_slo_ms=50,
            workload_profile={"mean_input_tokens": 1000, "mean_output_tokens": 200, "concurrency": 10},
        ),
    )
    assert normalized[0].arguments["workload"] == {
        "model": "Qwen/Qwen3-0.6B",
        "isl": 1000,
        "osl": 200,
        "concurrency": 10,
        "ttftMs": 500,
        "tpotMs": 50,
    }


def test_agreeing_preference_constraints_flow_to_aic_and_slo_filter(monkeypatch):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, arguments):
            calls.append((name, arguments))
            if name == "get_cluster_overview":
                return _cluster_result()
            return {
                "candidates": [
                    *_aic_result(ttft_ms=600, tpot_ms=30)["candidates"],
                    *_aic_result(ttft_ms=200, tpot_ms=40)["candidates"],
                ]
            }

    class Provider:
        turn = 0

        async def complete_tool_turn(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={}),
                        AIToolCall(
                            id="aic",
                            name="search_candidates",
                            arguments={
                                **_aic_arguments(),
                                "workload": {"ttftMs": 900, "tpotMs": 900},
                            },
                        ),
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Fits the observed cluster.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )
    result = asyncio.run(
        generator.generate(
            "cluster-a",
            "Qwen/Qwen3-0.6B",
            PlanningFacts(
                0.7,
                32,
                4,
                ttft_slo_ms=500,
                tpot_slo_ms=50,
                workload_profile=WorkloadProfile(mean_input_tokens=1000, mean_output_tokens=200),
            ),
            "user requests have an ISL of 1000 and an OSL of 200, requiring a TTFT of <= 500ms "
            "and a TPOT of <= 50ms under 10 concurrent users.",
        )
    )

    assert calls[1][1]["workload"] == {
        "model": "Qwen/Qwen3-0.6B",
        "isl": 1000,
        "osl": 200,
        "concurrency": 10,
        "ttftMs": 500,
        "tpotMs": 50,
    }
    assert result.proposals.tool_trace[1]["arguments"]["workload"] == calls[1][1]["workload"]
    assert result.resolved.facts.workload_profile.concurrency == 10
    assert result.resolved.facts.ttft_slo_ms == 500
    assert result.resolved.facts.tpot_slo_ms == 50
    assert result.resolved.trace[1]["aic_predictions_filtered_by_slo"] == 1
    assert len(result.resolved.facts.aic_predictions) == 1
    assert result.resolved.facts.aic_predictions[0].ttft_ms == 200


def test_preference_only_supplies_workload_and_slo_for_aic():
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(0.7, 32, 4, workload_profile=WorkloadProfile()),
        "ISL of 1000 and OSL of 200; TTFT of <= 500ms, TPOT of <= 50ms under 10 concurrent users",
    )
    tool = AIToolCall(id="aic", name="search_candidates", arguments={"workload": {}})

    assert AICandidateGenerator._normalize_required_tool_arguments(
        [tool],
        "cluster-a",
        "Qwen/Qwen3-0.6B",
        facts,
    )[0].arguments["workload"] == {
        "model": "Qwen/Qwen3-0.6B",
        "isl": 1000,
        "osl": 200,
        "concurrency": 10,
        "ttftMs": 500,
        "tpotMs": 50,
    }
    assert facts.workload_profile.concurrency == 10


def test_preference_natural_language_upper_limits_constrain_aic_targets():
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(0.7, 32, 4),
        "TTFT less than 40, TPOT less than 10",
    )
    tool = AIToolCall(id="aic", name="search_candidates", arguments={"workload": {"ttftMs": 50, "tpotMs": 20}})

    assert facts.ttft_slo_ms == 40
    assert facts.tpot_slo_ms == 10
    assert AICandidateGenerator._normalize_required_tool_arguments(
        [tool],
        "cluster-a",
        "Qwen/Qwen3-0.6B",
        facts,
    )[0].arguments["workload"] == {
        "model": "Qwen/Qwen3-0.6B",
        "isl": 4000,
        "osl": 1000,
        "ttftMs": 40,
        "tpotMs": 10,
    }


def test_preference_constraints_accept_legacy_dict_workload():
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(0.7, 32, 4, workload_profile={"mean_input_tokens": 4096}),
        "TTFT <= 500ms",
    )

    assert facts.workload_profile.mean_input_tokens == 4096
    assert facts.ttft_slo_ms == 500


def test_ai_generator_uses_aic_default_workload_when_none_is_provided(monkeypatch):
    calls = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, arguments):
            calls.append((name, arguments))
            return _cluster_result() if name == "get_cluster_overview" else _aic_result()

    class Provider:
        turn = 0

        async def complete_tool_turn(self, **_kwargs):
            self.turn += 1
            if self.turn > 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id="submit",
                            name="submit_candidate_proposals",
                            arguments={
                                "candidates": [
                                    {
                                        "provider_ref": "optimized-baseline",
                                        "replicas": 1,
                                        "tensor_parallel_size": 1,
                                        "rationale": "Fits the observed cluster.",
                                    }
                                ]
                            },
                        )
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(id="cluster", name="get_cluster_overview", arguments={}),
                    AIToolCall(
                        id="aic",
                        name="search_candidates",
                        arguments={
                            "workload": {"isl": 4096},
                            "searchConfig": {},
                            "sourceIds": ["aic"],
                        },
                    ),
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    asyncio.run(generator.generate("cluster-a", "Qwen/Qwen3-8B", PlanningFacts(8, 32, 4), None))

    assert calls[1] == (
        "search_candidates",
        {
            "workload": {"model": "Qwen/Qwen3-8B", "isl": 4000, "osl": 1000},
            "searchConfig": {"totalGpus": 4},
            "sourceIds": ["aic"],
        },
    )


def test_ai_generator_retries_an_invalid_candidate_submission(monkeypatch):
    requests = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            return _cluster_result() if name == "get_cluster_overview" else _aic_result()

    class Provider:
        turn = 0

        async def complete_tool_turn(self, **kwargs):
            self.turn += 1
            requests.append(kwargs["messages"])
            if self.turn == 1:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ]
                )
            if self.turn == 2:
                return AIToolTurn(
                    tool_calls=[
                        AIToolCall(
                            id="invalid",
                            name="submit_candidate_proposals",
                            arguments={},
                        )
                    ]
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="valid",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Fits the observed cluster.",
                                }
                            ]
                        },
                    )
                ]
            )

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    result = asyncio.run(
        generator.generate(
            "cluster-a",
            "Qwen/Qwen3-8B",
            PlanningFacts(8, 32, 4),
            None,
        )
    )

    assert result.proposals.candidates[0].provider_ref == "optimized-baseline"
    correction = requests[2][-1]["results"][0]
    assert correction["tool_call_id"] == "invalid"
    assert "submission schema invalid" in correction["content"]


def test_resolve_mcp_facts_discards_aic_predictions_that_fail_or_cannot_prove_slo():
    requested = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=1,
        free_gpu_count=0,
        ttft_slo_ms=500,
        tpot_slo_ms=50,
    )
    aic_result = {
        "candidates": [
            *_aic_result(ttft_ms=600, tpot_ms=30)["candidates"],
            *_aic_result(ttft_ms=300, tpot_ms=None)["candidates"],
            *_aic_result(ttft_ms=300, tpot_ms=30)["candidates"],
        ]
    }

    resolved = AICandidateGenerator._resolve_mcp_facts(
        "cluster-a",
        requested,
        {"get_cluster_overview": [_cluster_result()], "search_candidates": [aic_result]},
    )

    assert len(resolved.facts.aic_predictions) == 1
    assert resolved.facts.aic_predictions[0].ttft_ms == 300
    assert resolved.facts.aic_predictions[0].tpot_ms == 30
    assert "aic:search" in resolved.facts.evidence_ids


def test_ai_generator_prompt_encodes_candidate_priorities_and_tool_policy():
    prompt = AICandidateGenerator._system_prompt()

    assert "First request only get_cluster_overview and search_candidates" in prompt
    assert "deployment_intent.requirements is the shared normalized requirements contract" in prompt
    assert "never reinterpret raw preference text to override validated facts" in prompt
    assert "Conflicting form/text requirements require confirmation, not a silent choice" in prompt
    assert "operator_preference as the primary recommendation criterion" in prompt
    assert prompt.index("explicit operator requirements") < prompt.index("Provider semantics:")
    assert "Never relax an explicit operator requirement" in prompt
    assert "explicit operator requirement, or explicit SLO" in prompt
    assert "Diversity never requires an operator-forbidden provider or topology" in prompt
    assert "valid AIC predictions as secondary recommendation evidence" in prompt
    assert "Only when that search returns zero AIC candidates" in prompt
    assert "scale replicas first (prefill for TTFT, decode for TPOT in PD), then supported TP" in prompt
    assert "do not shrink below the anchor or exceed GPU/VRAM limits" in prompt
    assert "EP" not in prompt
    assert "Except for AIC-grounded scale-up proposals under rule 3" in prompt
    assert "unverified, not an AIC prediction or SLO guarantee" in prompt
    assert "deployment_intent.use_case and its resolved workload signals" in prompt
    assert "code-generation or a high shared_prefix_ratio" not in prompt
    assert "prefill_heavy workloads, prefer pd-disaggregation" not in prompt
    assert "historical configuration references, not current deployment SLO proof" in prompt
    assert "mapping_basis=source_label" in prompt
    assert "Do not describe the technique mapping as unverified" in prompt
    assert "Configuration equivalence and performance transfer still require validation" in prompt
    assert "unverified_source_hint" not in prompt
    assert "never combine different cases or series into synthetic measurements" in prompt
    assert "materially diverse candidates" in prompt
    assert "each feasible provider_ref" in prompt
    assert "GPU memory pressure" in prompt
    assert "CPU capacity" in prompt
    assert "minimum-resource topology" in prompt
    assert "lowest valid tensor parallelism and one replica per role" in prompt
    assert "If the initial search is empty, use the relaxed anchor" in prompt
    assert "operator_preference may select a provider but does not justify additional resources by itself" in prompt
    assert "Relaxed-anchor scaling remains unverified" in prompt
    assert "When explicit_workload is empty, do not use workload as an applicability gate" in prompt
    assert "Work backward from published TTFT/TPOT observations and their associated configurations" in prompt
    assert "cited public TTFT/TPOT configuration evidence supports additional resources" in prompt
    assert "shared_prefix_ratio of zero and the general-chat use case do not establish zero prefix reuse" in prompt
    assert "baseline-vllm" not in prompt
    assert "optimized-baseline: vLLM with llm-d prefix-affinity" in prompt
    assert "pd-disaggregation: separate prefill and decode roles" in prompt
    assert "tiered-prefix-cache: vLLM with tiered prefix caching that uses available CPU buffer" in prompt
    assert "to relieve GPU VRAM pressure" in prompt
    assert "Use tiered-prefix-cache only when the reported CPU buffer meets required_cpu_buffer_gib" in prompt
    assert "only the provided allowlisted MCP tools" in prompt


def test_ai_generator_emits_sanitized_progress_events(monkeypatch):
    events = []

    class Tools:
        async def available_tools(self):
            return [_cluster_tool(), _aic_tool()]

        async def call(self, name, _arguments):
            return _cluster_result() if name == "get_cluster_overview" else {"candidates": []}

    class Provider:
        turns = 0

        async def complete_tool_turn(self, **_kwargs):
            self.turns += 1
            if self.turns == 1:
                return AIToolTurn(
                    content="private reasoning",
                    tool_calls=[
                        AIToolCall(id="cluster", name="get_cluster_overview", arguments={"clusterId": "cluster-a"}),
                        AIToolCall(id="aic", name="search_candidates", arguments=_aic_arguments()),
                    ],
                )
            return AIToolTurn(
                tool_calls=[
                    AIToolCall(
                        id="submit",
                        name="submit_candidate_proposals",
                        arguments={
                            "candidates": [
                                {
                                    "provider_ref": "optimized-baseline",
                                    "replicas": 1,
                                    "tensor_parallel_size": 1,
                                    "rationale": "Small valid candidate.",
                                }
                            ]
                        },
                    )
                ]
            )

    async def on_progress(event):
        events.append(event)

    monkeypatch.setattr("llm_d_lens.agentic.generator.client_for", lambda **_kwargs: Provider())
    generator = AICandidateGenerator(
        OpenAIPlannerSettings(base_url="http://provider", model="planner"),
        tools=Tools(),
    )

    asyncio.run(
        generator.generate(
            "cluster-a",
            "Qwen/Qwen3-8B",
            PlanningFacts(8, 32, 4),
            None,
            on_progress=on_progress,
        )
    )

    assert [event["phase"] for event in events] == [
        "ai",
        "ai",
        "tools",
        "tools",
        "tools",
        "ai",
        "generation",
    ]
    assert events[1]["message"] == "AI requested evidence from get_cluster_overview, search_candidates."
    assert {events[3]["tool"], events[4]["tool"]} == {"get_cluster_overview", "search_candidates"}
    assert all("private reasoning" not in event["message"] for event in events)
    assert all("arguments" not in event for event in events)
