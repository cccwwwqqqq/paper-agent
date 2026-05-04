from __future__ import annotations

from typing import Literal

from langgraph.types import Send

from agentic_rag.agents.graph_state import AgentState, State


def route_after_rewrite(state: State):
    if not state.get("questionIsClear", False):
        return "request_clarification"

    intent_type = state.get("intent_type", "general_retrieval")
    task_intent = state.get("task_intent", "single_paper_qa")
    workspace_context = state.get("workspace_context", {}) or {}
    has_single_doc_target = bool(workspace_context.get("focus_paper_id") or (state.get("referenced_documents") or []))
    if task_intent == "workspace_memory_qa":
        return "workspace_memory_qa"
    if task_intent == "metadata_query":
        return "metadata_query_response"
    if task_intent == "single_paper_summary" and has_single_doc_target:
        return "single_paper_summary"
    if intent_type == "single_doc_close_reading" and has_single_doc_target:
        return "close_reading"
    if intent_type == "cross_doc_comparison":
        return "compare_papers"
    if intent_type == "literature_review":
        return "literature_review"
    if state.get("workspace_inventory_query"):
        return "workspace_inventory_response"

    return [
        Send(
            "agent",
            {
                "question": query,
                "question_index": idx,
                "messages": [],
                "workspace_context": state.get("workspace_context", {}),
                "working_memory": state.get("working_memory", {}),
                "memory_context": state.get("memory_context", {}),
                "memory_hits": state.get("memory_hits", []),
                "intent_type": intent_type,
                "task_intent": task_intent,
                "target_papers": state.get("target_papers", []),
                "retrieval_plan": state.get("retrieval_plan", {}),
                "answer_format": state.get("answer_format", "short_answer"),
                "intent_confidence": state.get("intent_confidence", 1.0),
                "artifact_kind": state.get("artifact_kind", ""),
            },
        )
        for idx, query in enumerate(state["rewrittenQuestions"])
    ]


def route_after_orchestrator_call(
    state: AgentState,
    *,
    max_iterations: int,
    max_tool_calls: int,
) -> Literal["tools", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= max_iterations or tool_count > max_tool_calls:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"

    return "tools"
