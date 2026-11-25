"""Conversation orchestration with explicit approval and terminal authorization failures."""

from __future__ import annotations

import asyncio
import json
import math
import re
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_d_lens.ai_providers.service import default_service as default_ai_provider_service
from llm_d_lens.assistant_tools.catalog import catalog
from llm_d_lens.assistant_tools.tools import api_request, invoke_tool

MAX_TOOL_CALL_TURNS = 40
MCP_TOOL_CALL_TIMEOUT = 630
AGENTIC_PLAN_TOOL = "create_agentic_plan"
DEPLOYMENT_NAME_TOOLS = {AGENTIC_PLAN_TOOL, "create_standard_vllm_deployment_run"}
SHORT_UUID_SUFFIX = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)
router = APIRouter(include_in_schema=False)
SYSTEM_PROMPT = (
    "When you start a deployment (or any long-running Lens operation) and check its status, if the "
    'status/pods look "pending", "queued", or "unschedulable" (e.g. a Kubernetes pod phase of Pending, or '
    "wait_for_status returning pending: true), tell the user right away that it looks resource-constrained "
    "and the request is queued -- do not silently keep waiting for it to finish. Only keep polling if the "
    "user asks you to keep monitoring.\n\n"
    "While a deployment is otherwise actively starting up (not stuck pending) and you are monitoring it, do "
    "not silently poll wait_for_status once with a long timeout and go quiet until it finishes. Instead, poll "
    "with a short timeoutSeconds (roughly 30-60s) each call, and every time you are about to poll again, "
    'include a brief plain-language status update in that same reply alongside the tool call -- e.g. "still '
    'starting up, 1/3 pods ready, checking again..." -- so the user sees periodic progress instead of one '
    "long silence followed by a final answer. Keep each update short; only stop polling and give your final "
    "answer once the deployment reaches a terminal state (ready, succeeded, failed, etc.) or the user asked "
    "you to stop.\n\n"
    "Authorization errors are final: if a tool returns an HTTP 401 or 403, or a body whose code is "
    '"forbidden", "unauthenticated" or "password_change_required", or a detail like "Missing permission: X" '
    'or "permission denied", do NOT retry, do NOT try other tools or arguments, and never claim the action '
    "succeeded. Stop immediately and tell the user plainly that they lack the required permission (quote the "
    "permission code when one is given) and suggest asking an administrator. This overrides the general "
    "error-recovery guidance below.\n\n"
    "If a tool call fails or comes back with an error and it is NOT an authorization error, do not just give "
    "up and report a raw error. First think about whether a different tool, different arguments, or a "
    "different order of operations could still accomplish what the user asked, and try that. Only if you "
    "genuinely cannot make progress should you stop -- and in that case, explain in plain language what you "
    "were trying to do, what went wrong, and include the actual error message so the user (or their operator) "
    "can act on it, instead of just surfacing a bare failure.\n\n"
    "When the user asks you to deploy a model, prefer the Agentic Deployment tools (create_agentic_plan, "
    "list_agentic_plans, get_agentic_plan, select_agentic_plan_candidate, refine_agentic_plan, "
    "approve_agentic_plan) over starting a deployment directly, since the planner can propose and score "
    "multiple candidate topologies instead of committing to just one guess. If the user states any "
    "preference or opinion about the deployment (e.g. prioritize latency vs. throughput, a specific replica "
    "count or parallelism, disaggregated prefill/decode, a GPU/hardware constraint, cost sensitivity, etc.), "
    "pass it along as the `planner_prompt` on create_agentic_plan (or refine_agentic_plan for an existing "
    "plan) so the planner actually accounts for it -- do not silently decide on the user's behalf. Once a "
    "plan comes back with candidates, present them to the user in plain language (key differences like "
    "replicas, tensor parallelism, prefill/decode split, and their supporting evidence) and ask which one "
    "they want -- do not pick a candidate yourself and proceed straight to approval. Only call "
    "select_agentic_plan_candidate after the user has told you which candidate they want, and only call "
    "approve_agentic_plan once the user has explicitly confirmed they want to deploy that selected "
    "candidate."
    "Deploying and publishing a Model Service are different steps. A ready deployment is not a published "
    "Model Service. When the user asks to publish or expose a model through the Gateway, create a Model "
    "Service, or make it callable through the shared Gateway, do not stop after the deployment becomes "
    "ready. Call list_publishable_deployments to confirm the ready execution, then publish_model_service "
    "with name, clusterId and executionId to create a new public model name, or create_model_service_member "
    "with groupId and executionId when the public model name already exists. Verify the result with "
    "get_model_service_group (or list_model_service_groups) and, when the user needs the client URL, "
    "get_model_service_connection. Only say a model service is published after the publish/member call "
    "succeeds and the group is queryable. If Gateway reconciliation is still in progress, say it is created "
    "and reconciling, not fully available. If the user did not specify a public model name, use a lowercase "
    "model-derived name such as qwen3-0.6b and tell the user which name you used; ask only if it conflicts "
    "or is ambiguous. When the request combines deploy and publish, complete both phases before giving the "
    "final answer."
)


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate_tokens(text) -> int:
    if not isinstance(text, str) or not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    return math.ceil(cjk + (len(text.encode("utf-16-le")) // 2 - cjk) / 4)


def conversation_tokens(conversation) -> int:
    return sum(
        estimate_tokens(message.get("content"))
        + estimate_tokens(message.get("reasoning_content"))
        + 4
        + sum(
            estimate_tokens(call["function"].get("name")) + estimate_tokens(call["function"].get("arguments"))
            for call in message.get("tool_calls") or []
        )
        for message in conversation
    )


def truncate_tool_results(conversation):
    for message in conversation:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and len(content) > 1500:
            message["content"] = content[:1500] + (
                f"\n...[truncated {len(content) - 1500} more characters to fit the model's context window]"
            )


def split_inline_think(content):
    if not content:
        return None, content
    match = re.search(r"<think>([\s\S]*?)</think>", content, re.IGNORECASE)
    if not match:
        return None, content
    return match[1].strip() or None, (content[: match.start()] + content[match.end() :]).strip()


def detect_permission_failure(result):
    text = (
        result
        if isinstance(result, str)
        else next((item.get("text") for item in result.get("content", []) if item.get("type") == "text"), None)
        if isinstance(result, dict)
        else None
    )
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    problem = payload["detail"] if isinstance(payload.get("detail"), dict) else payload
    status, code = problem.get("status", payload.get("status")), problem.get("code", payload.get("code"))
    detail = problem.get("detail", payload.get("detail", ""))
    if status == 401 or code == "unauthenticated":
        return {"kind": "unauthenticated"}
    if status != 403 or code == "csrf_failed":
        return None
    if code == "password_change_required":
        return {"kind": "password_change_required"}
    detail = detail if isinstance(detail, str) else ""
    if "cluster is not accessible" in detail.lower():
        return {"kind": "cluster_not_accessible"}
    match = re.search(r"Missing permission:\s*([^\s\"']+)", detail, re.IGNORECASE)
    return {"kind": "forbidden", **({"requiredPermission": match[1]} if match else {})}


def permission_message(name, failure):
    if failure["kind"] == "unauthenticated":
        return "Your session has expired. Please sign in again."
    if failure["kind"] == "password_change_required":
        return "You must change your password before Lens can run that action."
    if failure["kind"] == "cluster_not_accessible":
        return f'You do not have access to the cluster required by "{name}".'
    if failure.get("requiredPermission"):
        return (
            f'You do not have permission to run "{name}". Missing permission: {failure["requiredPermission"]}. '
            "Ask an administrator to grant it."
        )
    return f'You do not have permission to run "{name}" (permission denied). Ask an administrator.'


def tool_arguments(call):
    try:
        return json.loads(call["function"].get("arguments") or "{}")
    except ValueError:
        return {}


def append_tool_result(conversation, call, result):
    conversation.append({"role": "tool", "tool_call_id": call["id"], "content": encode(result)})


def prepare_tool_arguments(name, arguments, provider_id):
    """Carry the assistant's external provider into an Agentic planning call.

    The assistant model cannot reliably know which saved provider connection is
    serving it, so leaving ``ai_provider_id`` to the tool call makes Agentic
    Deployment fall back to deterministic planning. Only fill an absent/blank
    value; an explicit provider chosen by the model still wins.
    """
    prepared = arguments
    if name == AGENTIC_PLAN_TOOL and provider_id and not str(prepared.get("ai_provider_id") or "").strip():
        prepared = {**prepared, "ai_provider_id": provider_id}
    if name in DEPLOYMENT_NAME_TOOLS:
        deployment_name = str(prepared.get("deployment_name") or "").strip()
        if deployment_name and not SHORT_UUID_SUFFIX.search(deployment_name):
            # Keep assistant-created deployments unique. The backend request
            # caps this field at 120 characters, so reserve room for "-<uuid8>".
            base = deployment_name.rstrip("-")[:111] or "deployment"
            prepared = {**prepared, "deployment_name": f"{base}-{uuid4().hex[:8]}"}
    return prepared


async def call_tool_safely(request, name, arguments):
    try:
        async with asyncio.timeout(MCP_TOOL_CALL_TIMEOUT):
            result = await invoke_tool(request, name, arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False}, False
    except Exception as error:
        return {"isError": True, "error": str(error)}, True


async def process_pending_tool_calls(
    tool_calls, conversation, request, emit, read_only=False, default_ai_provider_id=None
):
    definitions = {tool["name"]: tool for tool in catalog(request.app)}
    for index, call in enumerate(tool_calls):
        name = call["function"]["name"]
        arguments = prepare_tool_arguments(name, tool_arguments(call), default_ai_provider_id)
        definition = definitions.get(name)
        if definition is None:
            await emit("blocked", {"name": name, "arguments": arguments, "reason": "unknown tool"})
            append_tool_result(conversation, call, {"error": "blocked: unknown tool"})
            continue
        if read_only and definition["riskTier"] != "read":
            await emit(
                "blocked",
                {
                    "name": name,
                    "arguments": arguments,
                    "reason": "this turn started from voice input and is restricted to read-only tools",
                },
            )
            append_tool_result(
                conversation, call, {"error": "blocked: voice-originated turns cannot run write/approve-tier tools"}
            )
            continue
        if definition["riskTier"] == "approve":
            await emit(
                "confirm_required",
                {
                    "id": call["id"],
                    "name": name,
                    "arguments": arguments,
                    "riskTier": "approve",
                    "description": definition["description"],
                    "resumeConversation": conversation,
                },
            )
            return "paused"
        await emit("tool_call", {"id": call["id"], "name": name, "arguments": arguments})
        result, is_error = await call_tool_safely(request, name, arguments)
        await emit("tool_result", {"id": call["id"], "name": name, "result": result, "isError": is_error})
        append_tool_result(conversation, call, result)
        if failure := detect_permission_failure(result):
            await emit(
                "permission_denied",
                {"id": call["id"], "name": name, **failure, "message": permission_message(name, failure)},
            )
            for sibling in tool_calls[index + 1 :]:
                append_tool_result(
                    conversation, sibling, {"error": "not executed: a previous tool call was denied by permissions"}
                )
            return "denied"
    return "continue"


def hf_context_length(config):
    if not isinstance(config, dict):
        return None
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length", "model_max_length"):
        value = config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 10**9:
            return int(value)
    for key in ("text_config", "llm_config", "language_config"):
        if found := hf_context_length(config.get(key)):
            return found
    return None


async def discover_max_model_len(model):
    metadata = Path(__file__).resolve().parents[5] / "packages/shared/src/catalog/modelCatalogMetadata.json"
    if metadata.is_file():
        entries = json.loads(metadata.read_text())
        entry = next((value for key, value in entries.items() if key.lower() == model.lower()), {})
        if entry.get("contextLength"):
            return entry["contextLength"]
    try:
        async with httpx.AsyncClient(timeout=4, follow_redirects=True) as client:
            repository = model
            if "/" not in repository:
                response = await client.get("https://huggingface.co/api/models", params={"search": model, "limit": 1})
                response.raise_for_status()
                repository = response.json()[0]["id"]
            response = await client.get(
                f"https://huggingface.co/{quote(repository, safe='/')}/resolve/main/config.json"
            )
            response.raise_for_status()
            return hf_context_length(response.json()) or 4096
    except (httpx.HTTPError, ValueError, IndexError, KeyError):
        return 4096


async def call_provider(request, deployment, conversation, tools):
    body = {"model": deployment["model"], "messages": conversation}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    result = await api_request(
        request, f"/api/v1/ai-providers/{quote(deployment['providerId'], safe='')}/chat/completions", "POST", body
    )
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(f"External AI provider request failed: {encode(result)[:500]}")
    return result


async def resume_pending(request, resume, conversation, emit, read_only, default_ai_provider_id=None):
    assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant" and message.get("tool_calls")
        ),
        None,
    )
    call = next((item for item in (assistant or {}).get("tool_calls", []) if item["id"] == resume["toolCallId"]), None)
    if call is None or any(message.get("tool_call_id") == call["id"] for message in conversation):
        raise ValueError("Could not find the pending tool call to resolve -- it may have already been handled.")
    name = call["function"]["name"]
    definition = next((tool for tool in catalog(request.app) if tool["name"] == name), None)
    if resume["decision"] == "approve" and definition and definition["riskTier"] == "approve" and not read_only:
        result, is_error = await call_tool_safely(request, name, tool_arguments(call))
        await emit("tool_result", {"id": call["id"], "name": name, "result": result, "isError": is_error})
    else:
        result = {
            "rejected": True,
            "reason": (
                "this turn started from voice input and is restricted to read-only tools"
                if read_only
                else "this tool is no longer registered or is not approvable and was not executed"
                if resume["decision"] == "approve"
                else "the user rejected this action; it was not executed"
            ),
        }
        await emit("tool_result", {"id": call["id"], "name": name, "result": result})
    append_tool_result(conversation, call, result)
    siblings = [
        item
        for item in assistant["tool_calls"]
        if not any(message.get("tool_call_id") == item["id"] for message in conversation)
    ]
    if failure := detect_permission_failure(result):
        await emit(
            "permission_denied",
            {"id": call["id"], "name": name, **failure, "message": permission_message(name, failure)},
        )
        for sibling in siblings:
            append_tool_result(
                conversation, sibling, {"error": "not executed: a previous tool call was denied by permissions"}
            )
        return "denied"
    return await process_pending_tool_calls(
        siblings, conversation, request, emit, read_only, default_ai_provider_id=default_ai_provider_id
    )


async def run_turn(request, body, emit):
    deployment = body["deployment"]
    read_only = bool(body.get("readOnly"))
    assistant_provider_id = deployment.get("providerId") or None
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": f"[{tool['riskTier']}] {tool['description']}",
                "parameters": tool["inputSchema"],
            },
        }
        for tool in catalog(request.app)
        if not read_only or tool["riskTier"] == "read"
    ]
    provider = default_ai_provider_service().get(deployment.get("providerId", ""))
    max_length = (provider.context_window if provider is not None else None) or await discover_max_model_len(
        deployment["model"]
    )
    budget = math.floor(max_length * 0.8)
    resume = body.get("pendingResume")
    conversation = deepcopy(resume["conversation"] if resume else body["messages"])
    if conversation[0].get("role") != "system":
        conversation.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    outcome = (
        await resume_pending(
            request, resume, conversation, emit, read_only, default_ai_provider_id=assistant_provider_id
        )
        if resume
        else "continue"
    )
    final_answer, turn = None, 0
    while outcome == "continue" and turn < MAX_TOOL_CALL_TURNS:
        turn += 1
        tools_tokens = estimate_tokens(encode(tools))
        if tools_tokens + conversation_tokens(conversation) > budget:
            truncate_tool_results(conversation)
        await emit(
            "context_usage", {"maxModelLen": max_length, "usedTokens": tools_tokens + conversation_tokens(conversation)}
        )
        completion = await call_provider(request, deployment, conversation, tools)
        choices = completion.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not message:
            raise ValueError("Deployment returned no message")
        if message.get("reasoning_content"):
            await emit("think", {"content": message["reasoning_content"]})
        think, answer = split_inline_think(message.get("content"))
        if think:
            await emit("think", {"content": think})
        if not message.get("tool_calls"):
            final_answer = answer or ""
            await emit("message", {"content": final_answer})
            break
        if answer:
            await emit("progress", {"content": answer})
        conversation.append(message)
        outcome = await process_pending_tool_calls(
            message["tool_calls"],
            conversation,
            request,
            emit,
            read_only,
            default_ai_provider_id=assistant_provider_id,
        )
    if outcome == "continue" and final_answer is None and turn >= MAX_TOOL_CALL_TURNS:
        conversation.append(
            {
                "role": "system",
                "content": (
                    "You have reached the maximum number of tool-calling steps for this turn. Do not "
                    "attempt to call any more tools -- none will be executed. Summarize, for the user: "
                    "what you have done so far, what you found, what (if anything) remains unresolved, "
                    "and what you recommend they do next."
                ),
            }
        )
        completion = await call_provider(request, deployment, conversation, [])
        message = (completion.get("choices") or [{}])[0].get("message", {})
        if message.get("reasoning_content"):
            await emit("think", {"content": message["reasoning_content"]})
        think, answer = split_inline_think(message.get("content"))
        if think:
            await emit("think", {"content": think})
        final_answer = answer or ""
        await emit("message", {"content": final_answer})
    if final_answer is not None:
        history = [
            message
            for message in conversation
            if message.get("role") == "user" or (message.get("role") == "assistant" and not message.get("tool_calls"))
        ]
        history.append({"role": "assistant", "content": final_answer})
        tools_tokens = estimate_tokens(encode(tools))
        before = tools_tokens + conversation_tokens(history)
        if before > budget:
            await emit("compaction_start", {"usedTokens": before, "maxModelLen": max_length})
            dropped = 0
            while len(history) > 2 and tools_tokens + conversation_tokens(history) > budget:
                history.pop(0)
                dropped += 1
            await emit(
                "compaction_end",
                {
                    "droppedMessageCount": dropped,
                    "usedTokensBefore": before,
                    "usedTokensAfter": tools_tokens + conversation_tokens(history),
                    "maxModelLen": max_length,
                },
            )
    await emit("done", {})


@router.post("/api/playground/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    body = body if isinstance(body, dict) else {}
    deployment = body.get("deployment") or {}
    if (
        not isinstance(deployment, dict)
        or bool(deployment.get("executionId")) == bool(deployment.get("providerId"))
        or not deployment.get("model")
    ):
        return JSONResponse(
            {
                "error": (
                    "Exactly one of deployment.executionId or deployment.providerId (plus deployment.model) is required"
                )
            },
            status_code=400,
        )
    if deployment.get("executionId"):
        return JSONResponse(
            {"error": "Lens Assistant does not support internal deployments yet. Use an External AI provider."},
            status_code=400,
        )
    resume = body.get("pendingResume")
    if resume:
        if (
            not isinstance(resume, dict)
            or not isinstance(resume.get("conversation"), list)
            or not resume["conversation"]
            or not resume.get("toolCallId")
            or resume.get("decision") not in {"approve", "reject"}
        ):
            return JSONResponse(
                {
                    "error": (
                        "pendingResume must include a non-empty conversation, a toolCallId, "
                        'and decision "approve" or "reject"'
                    )
                },
                status_code=400,
            )
    elif not isinstance(body.get("messages"), list) or not body["messages"]:
        return JSONResponse({"error": "messages must be a non-empty array"}, status_code=400)

    async def stream():
        events = asyncio.Queue(maxsize=32)

        async def emit(event, data):
            await events.put(f"event: {event}\ndata: {encode(data)}\n\n")

        async def produce():
            try:
                await run_turn(request, body, emit)
            except Exception as error:
                await emit("error", {"message": str(error)})
            finally:
                if not asyncio.current_task().cancelling():
                    await events.put(None)

        task = asyncio.create_task(produce())
        try:
            while (event := await events.get()) is not None:
                yield event
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
