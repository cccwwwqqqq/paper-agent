from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Set

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from agentic_rag.agents.schemas import AnswerVerification, IntentAnalysis, PaperProfile
from agentic_rag.agents.retriever_policy import RetrieverPolicy
from agentic_rag.utils import estimate_context_tokens

from .graph_state import AgentState, State
from .prompts import (
    get_aggregation_prompt,
    get_close_reading_prompt,
    get_comparison_prompt,
    get_context_compression_prompt,
    get_conversation_summary_prompt,
    get_fallback_response_prompt,
    get_literature_review_prompt,
    get_metadata_query_prompt,
    get_orchestrator_prompt,
    get_paper_profile_prompt,
    get_reflection_prompt,
    get_rewrite_query_prompt,
    get_single_paper_summary_prompt,
    get_verification_prompt,
    get_workspace_memory_qa_prompt,
)


def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {
            "conversation_summary": "",
            "retrieved_chunks": ["__reset__"],
            "retrieved_parent_chunks": ["__reset__"],
            "retrieved_paper_ids": ["__reset__"],
            "retrieved_sections": ["__reset__"],
        }

    relevant_msgs = [
        msg
        for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {
            "conversation_summary": "",
            "retrieved_chunks": ["__reset__"],
            "retrieved_parent_chunks": ["__reset__"],
            "retrieved_paper_ids": ["__reset__"],
            "retrieved_sections": ["__reset__"],
        }

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke(
        [SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)]
    )
    working_memory = state.get("working_memory", {}) or {}
    working_memory["recent_summary"] = summary_response.content
    return {
        "conversation_summary": summary_response.content,
        "working_memory": working_memory,
        "agent_answers": [{"__reset__": True}],
        "paper_profiles": [],
        "retrieved_chunks": ["__reset__"],
        "retrieved_parent_chunks": ["__reset__"],
        "retrieved_paper_ids": ["__reset__"],
        "retrieved_sections": ["__reset__"],
    }


def _normalize_intent_response(response):
    if response is None:
        return None
    if isinstance(response, IntentAnalysis):
        return response
    if isinstance(response, dict):
        try:
            return IntentAnalysis.model_validate(response)
        except Exception:
            return None
    return None


def _normalize_lines(text: str) -> list[str]:
    return [line.rstrip() for line in str(text or "").splitlines()]


def _parse_structured_tool_block(block: str) -> dict:
    labels = {
        "Rerank Backend:": "rerank_backend",
        "Rank Score:": "rank_score",
        "Parent ID:": "parent_id",
        "Paper ID:": "paper_id",
        "Section:": "section",
        "Pages:": "pages",
        "Content Type:": "content_type",
        "File Name:": "source",
        "Source:": "source",
        "Content:": "content",
    }
    parsed = {key: "" for key in labels.values()}
    current_key = None

    for raw_line in _normalize_lines(block):
        line = raw_line.strip()
        matched_key = None
        for prefix, key in labels.items():
            if line.startswith(prefix):
                parsed[key] = line[len(prefix):].strip()
                current_key = key
                matched_key = key
                break
        if matched_key or not raw_line.strip():
            continue
        if current_key == "content":
            parsed[current_key] = f"{parsed[current_key]}\n{raw_line}".strip()

    return parsed


def _parse_search_tool_content(content: str) -> list[dict]:
    text = str(content or "").strip()
    if not text or text in {"NO_RELEVANT_CHUNKS"} or text.startswith("RETRIEVAL_ERROR:"):
        return []

    blocks = []
    current = []
    for line in _normalize_lines(text):
        if line.startswith("Parent ID:") and current:
            blocks.append("\n".join(current))
            current = [line]
        elif line.startswith("Rerank Backend:") and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    records = []
    for block in blocks:
        parsed = _parse_structured_tool_block(block)
        if parsed.get("parent_id"):
            records.append(parsed)
    return records


def _parse_parent_tool_content(content: str) -> list[dict]:
    text = str(content or "").strip()
    if not text or text in {"NO_PARENT_DOCUMENT"} or text.startswith("PARENT_RETRIEVAL_ERROR:"):
        return []
    parsed = _parse_structured_tool_block(text)
    return [parsed] if parsed.get("parent_id") else []


def _record_from_doc(doc) -> dict:
    metadata = doc.metadata or {}
    return {
        "parent_id": metadata.get("parent_id", ""),
        "paper_id": metadata.get("paper_id", ""),
        "section": metadata.get("section", ""),
        "pages": f"{metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}",
        "source": metadata.get("source", ""),
        "content_type": metadata.get("content_type", "paragraph"),
        "content": doc.page_content.strip(),
    }


def _record_from_parent_chunk(item: dict) -> dict:
    metadata = item.get("metadata", {}) or {}
    return {
        "parent_id": item.get("parent_id", ""),
        "paper_id": metadata.get("paper_id", ""),
        "section": metadata.get("section", ""),
        "pages": f"{metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}",
        "source": metadata.get("source", ""),
        "content_type": metadata.get("content_type", "paragraph"),
        "content": item.get("content", "").strip(),
    }


def _unique_non_empty(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _extract_retrieval_trace_from_messages(messages) -> tuple[list[dict], list[dict], list[str], list[str]]:
    retrieved_chunks = []
    retrieved_parent_chunks = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", "") == "search_child_chunks":
            retrieved_chunks.extend(_parse_search_tool_content(message.content))
        elif getattr(message, "name", "") == "retrieve_parent_chunks":
            retrieved_parent_chunks.extend(_parse_parent_tool_content(message.content))

    retrieved_paper_ids = _unique_non_empty(
        [item.get("paper_id", "") for item in retrieved_chunks + retrieved_parent_chunks]
    )
    retrieved_sections = _unique_non_empty(
        [item.get("section", "") for item in retrieved_chunks + retrieved_parent_chunks]
    )
    return retrieved_chunks, retrieved_parent_chunks, retrieved_paper_ids, retrieved_sections


def _extract_rerank_backend_from_messages(messages) -> str:
    for message in messages:
        if not isinstance(message, ToolMessage) or getattr(message, "name", "") != "search_child_chunks":
            continue
        records = _parse_search_tool_content(message.content)
        for item in records:
            backend = str(item.get("rerank_backend", "")).strip()
            if backend:
                return backend
    return "heuristic_fallback"


def _format_memory_hits(memory_hits: list[dict]) -> str:
    if not memory_hits:
        return "No relevant memory hits found."
    formatted = []
    for item in memory_hits:
        source = item.get("source", "memory")
        content = item.get("content", "")
        formatted.append(f"- [{source}] {content}")
    return "\n".join(formatted)


def _dialogue_paper_id(working_memory: dict, workspace_context: dict, episodic_hits: list[dict] | None = None) -> str | None:
    if working_memory.get("current_dialogue_paper_id"):
        return working_memory["current_dialogue_paper_id"]

    recent_papers = [paper_id for paper_id in (working_memory.get("recent_papers", []) or []) if paper_id]
    if recent_papers:
        return recent_papers[0]

    for item in episodic_hits or []:
        for paper_id in item.get("paper_ids", []) or []:
            if paper_id:
                return paper_id

    return working_memory.get("focus_paper_id") or workspace_context.get("focus_paper_id")


def _heuristic_intent_analysis(query: str, workspace_context: dict, working_memory: dict, dialogue_paper_id: str | None = None):
    lowered = (query or "").lower()
    focus_paper_id = working_memory.get("focus_paper_id") or workspace_context.get("focus_paper_id")
    pronoun_reference = _refers_to_current_paper(lowered)
    previous_paper_reference = _refers_to_previous_paper(lowered)
    workspace_scope_query = _is_workspace_scope_query(lowered)
    workspace_inventory_query = _looks_like_workspace_inventory_query(lowered)
    referenced_documents = []
    if workspace_scope_query:
        referenced_documents = []
    elif previous_paper_reference and working_memory.get("recent_papers"):
        recent_papers = [paper for paper in working_memory.get("recent_papers", []) if paper]
        current_paper = dialogue_paper_id or focus_paper_id
        referenced_documents = [paper for paper in recent_papers if paper != current_paper][:1]
    elif pronoun_reference and dialogue_paper_id:
        referenced_documents = [dialogue_paper_id]
    elif focus_paper_id:
        referenced_documents = [focus_paper_id]
    elif working_memory.get("recent_papers"):
        recent_papers = [paper for paper in working_memory.get("recent_papers", []) if paper]
        if pronoun_reference:
            referenced_documents = recent_papers[:1]

    if any(token in lowered for token in ["literature review", "survey", "\u7efc\u8ff0"]) or (
        any(token in lowered for token in ["summarize", "synthesize", "review"])
        and any(token in lowered for token in ["papers", "workspace", "across", "three papers", "multiple papers"])
    ):
        intent_type = "literature_review"
    elif _is_comparison_question(lowered):
        intent_type = "cross_doc_comparison"
    elif workspace_inventory_query:
        intent_type = "general_retrieval"
    elif referenced_documents or any(token in lowered for token in ["this paper", "the paper above", "\u8fd9\u7bc7\u8bba\u6587", "\u8be5\u8bba\u6587"]):
        intent_type = "single_doc_close_reading"
    else:
        intent_type = "general_retrieval"

    comparison_targets = referenced_documents if intent_type in {"cross_doc_comparison", "literature_review"} else []
    if intent_type == "literature_review" and workspace_scope_query:
        comparison_targets = []
    task_intent = _infer_task_intent(query, intent_type, workspace_inventory_query)
    return IntentAnalysis(
        intent_type=intent_type,
        task_intent=task_intent,
        resolved_query=query,
        rewritten_questions=[query],
        referenced_documents=referenced_documents,
        comparison_targets=comparison_targets,
        target_papers=comparison_targets or referenced_documents,
        topic="",
        question_type={
            "single_paper_summary": "summarize",
            "method_explanation": "explain",
            "multi_paper_comparison": "compare",
            "literature_review": "synthesize",
            "citation_finding": "find_evidence",
            "workspace_memory_qa": "memory_lookup",
            "metadata_query": "query_metadata",
        }.get(task_intent, "answer"),
        retrieval_scope=[],
        answer_format={
            "single_paper_summary": "structured_summary",
            "method_explanation": "structured_explanation",
            "multi_paper_comparison": "comparison_table" if _contains_any(lowered, ["table", "\u8868"]) else "short_answer",
            "citation_finding": "evidence_list",
            "workspace_memory_qa": "short_answer",
            "metadata_query": "metadata_list",
        }.get(task_intent, "short_answer"),
        need_memory=task_intent == "workspace_memory_qa",
        need_metadata_filter=task_intent == "metadata_query",
        confidence=0.75,
        retrieval_plan={},
        artifact_kind="",
        needs_clarification=False,
        clarification_question="",
    )


def _detect_artifact_kind(query: str, intent_type: str) -> str:
    lowered = str(query or "").lower()
    if _contains_any(lowered, ["对比表", "比较表", "comparison table", "table comparing", "markdown table"]):
        return "build_comparison_table"
    if _contains_any(lowered, ["笔记", "整理笔记", "study notes", "notes for", "export notes"]):
        return "export_notes"
    if _contains_any(lowered, ["研究摘要", "摘要卡片", "research summary", "summary card", "executive summary"]):
        return "save_research_summary"
    if intent_type == "literature_review" and _contains_any(lowered, ["总结", "summary"]) and "review" not in lowered:
        return "save_research_summary"
    return ""


def _refers_to_current_paper(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "it",
            "this paper",
            "that paper",
            "该论文",
            "这篇论文",
            "它",
        ],
    )


def _refers_to_previous_paper(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "the previous paper",
            "previous paper",
            "previous one",
            "\u4e0a\u4e00\u7bc7",
            "\u524d\u4e00\u7bc7",
            "\u4e0a\u4e00\u7bc7\u8bba\u6587",
            "\u524d\u4e00\u7bc7\u8bba\u6587",
        ],
    )


def _is_short_follow_up_query(text: str) -> bool:
    raw_text = str(text or "").strip()
    if not raw_text:
        return False
    lowered = raw_text.lower()
    normalized = _normalize_lookup_text(raw_text)
    token_count = len(normalized.split()) if normalized else 0
    return (
        len(raw_text) <= 24
        and token_count <= 6
        and (
            _refers_to_current_paper(lowered)
            or _refers_to_previous_paper(lowered)
            or _contains_any(lowered, ["what about", "\u5462", "\u90a3", "\u90a3\u7bc7"])
        )
    )


def _default_section_hints(query_profile: dict, intent_type: str) -> list[str]:
    hints = list(query_profile.get("explicit_section_hints", []) or [])
    if intent_type == "single_doc_close_reading":
        hints.extend(["method", "procedure", "system model", "construction"])
    elif intent_type == "cross_doc_comparison":
        if query_profile.get("is_performance"):
            hints.extend(["performance", "experimental", "evaluation", "results"])
        elif query_profile.get("is_contribution"):
            hints.extend(["contribution", "main idea", "conclusion"])
        else:
            hints.extend(["system model", "construction", "main idea", "contribution"])
    elif intent_type == "literature_review":
        if query_profile.get("is_performance"):
            hints.extend(["performance", "experimental", "evaluation"])
        elif query_profile.get("is_security"):
            hints.extend(["security", "threat model", "security model"])
        else:
            hints.extend(["contribution", "main idea", "conclusion"])
    return _unique_non_empty(hints)


def _task_section_hints(task_intent: str, retrieval_scope: list[str]) -> list[str]:
    hints = list(retrieval_scope or [])
    if task_intent == "single_paper_summary":
        hints.extend(["abstract", "introduction", "contribution", "method", "experiment", "result", "conclusion", "limitation"])
    elif task_intent == "method_explanation":
        hints.extend(["method", "algorithm", "formula", "construction", "architecture", "training objective", "loss"])
    elif task_intent == "citation_finding":
        hints.extend(["related work", "contribution", "main idea", "experiment", "result", "conclusion"])
    elif task_intent == "metadata_query":
        hints.extend(["metadata"])
    return _unique_non_empty(hints)


def _default_content_type_hints(query_profile: dict, intent_type: str) -> list[str]:
    hints = []
    if query_profile.get("is_performance"):
        hints.extend(["table", "figure", "caption", "paragraph"])
    elif query_profile.get("is_algorithmic") or query_profile.get("is_process"):
        hints.extend(["algorithm", "formula", "paragraph", "list"])
    elif query_profile.get("is_system") or query_profile.get("is_security"):
        hints.extend(["paragraph", "algorithm"])
    elif intent_type in {"cross_doc_comparison", "literature_review"}:
        hints.extend(["paragraph", "algorithm", "table"])
    return _unique_non_empty(hints or ["paragraph"])


def _default_answer_format(task_intent: str, proposed: dict) -> str:
    if proposed.get("answer_format"):
        return str(proposed["answer_format"])
    return {
        "single_paper_summary": "structured_summary",
        "method_explanation": "structured_explanation",
        "multi_paper_comparison": "comparison_table",
        "literature_review": "related_work_paragraph",
        "citation_finding": "evidence_list",
        "workspace_memory_qa": "short_answer",
        "metadata_query": "metadata_list",
    }.get(task_intent, "short_answer")


def _default_retrieval_scope(task_intent: str, proposed: dict) -> list[str]:
    if proposed.get("retrieval_scope"):
        return _unique_non_empty(proposed.get("retrieval_scope") or [])
    return {
        "single_paper_summary": ["abstract", "introduction", "method", "experiment", "conclusion", "limitation"],
        "method_explanation": ["method", "algorithm", "formula", "construction", "architecture", "training objective"],
        "citation_finding": ["full_paper", "related_work", "method", "experiment", "result"],
        "literature_review": ["abstract", "introduction", "method", "related_work", "conclusion"],
        "multi_paper_comparison": ["method", "experiment", "result", "limitation"],
        "workspace_memory_qa": ["workspace_memory"],
        "metadata_query": ["metadata"],
    }.get(task_intent, ["full_paper"])


def _build_retrieval_plan(
    *,
    query: str,
    intent_type: str,
    referenced_documents: list[str],
    comparison_targets: list[str],
    workspace_context: dict,
    paper_catalog: list[dict],
    task_intent: str | None = None,
    proposed_plan: dict | None = None,
) -> dict:
    query_profile = _extract_query_profile(query)
    proposed = proposed_plan or {}
    task_intent = _normalize_task_intent(
        task_intent or proposed.get("task_intent") or workspace_context.get("task_intent"),
        query,
        intent_type,
        workspace_inventory_query=bool(workspace_context.get("workspace_inventory_query")),
    )
    retrieval_scope = _default_retrieval_scope(task_intent, proposed)
    if intent_type == "single_doc_close_reading":
        mode = "single_doc"
        paper_ids = referenced_documents[:1]
        per_paper_limit = 8 if task_intent == "single_paper_summary" else (6 if task_intent in {"method_explanation", "citation_finding"} else 4)
        global_limit = per_paper_limit
    elif intent_type == "cross_doc_comparison":
        mode = "per_paper_compare"
        paper_ids = comparison_targets or referenced_documents
        per_paper_limit = 4 if task_intent == "citation_finding" else 3
        global_limit = max(len(paper_ids) * 3, 6) if paper_ids else 6
    elif intent_type == "literature_review":
        mode = "literature_review"
        paper_ids = comparison_targets or referenced_documents or [paper.get("paper_id", "") for paper in paper_catalog if paper.get("paper_id")]
        per_paper_limit = 4 if task_intent == "citation_finding" else 3
        global_limit = max(len(paper_ids) * 2, 6) if paper_ids else 6
    else:
        mode = "general"
        paper_ids = referenced_documents[:1]
        per_paper_limit = 3
        global_limit = 6

    proposed_paper_ids = _unique_non_empty(proposed.get("paper_ids") or [])
    if intent_type in {"cross_doc_comparison", "literature_review"}:
        resolved_paper_ids = _unique_non_empty(paper_ids + proposed_paper_ids)
    else:
        resolved_paper_ids = proposed_paper_ids or paper_ids

    final_plan = {
        "mode": proposed.get("mode") or mode,
        "task_intent": task_intent,
        "paper_ids": resolved_paper_ids,
        "section_hints": _unique_non_empty(
            proposed.get("section_hints")
            or (_default_section_hints(query_profile, intent_type) + _task_section_hints(task_intent, retrieval_scope))
        ),
        "content_type_hints": _unique_non_empty(proposed.get("content_type_hints") or _default_content_type_hints(query_profile, intent_type)),
        "retrieval_scope": retrieval_scope,
        "answer_format": _default_answer_format(task_intent, proposed),
        "need_memory": bool(proposed.get("need_memory", False) or task_intent == "workspace_memory_qa"),
        "need_metadata_filter": bool(proposed.get("need_metadata_filter", False) or task_intent == "metadata_query"),
        "per_paper_limit": int(proposed.get("per_paper_limit") or per_paper_limit),
        "global_limit": int(proposed.get("global_limit") or global_limit),
    }
    if workspace_context.get("focus_paper_id") and final_plan["mode"] == "single_doc" and not final_plan["paper_ids"]:
        final_plan["paper_ids"] = [workspace_context["focus_paper_id"]]
    return final_plan


def _build_memory_context(session_memory: dict, semantic_hits: list[dict], episodic_hits: list[dict]) -> dict:
    return {
        "session": session_memory or {},
        "semantic": [{k: v for k, v in item.items() if k != "_score"} for item in semantic_hits],
        "episodic": [{k: v for k, v in item.items() if k != "_score"} for item in episodic_hits],
    }


_GENERIC_PAPER_ALIASES = {"abe", "cp-abe", "kp-abe", "abac", "iot", "iiot"}
_SCHEME_ALIAS_PARTS = {"abe", "bac", "pbac", "abac"}


def _extract_query_alias_tokens(query: str) -> list[str]:
    query_text = str(query or "")
    return _unique_non_empty(
        [
            match.lower()
            for match in re.findall(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b", query_text.upper())
            if len(match) >= 5
        ]
    )


def _looks_like_scheme_alias_token(token: str) -> bool:
    parts = [part.lower() for part in str(token or "").split("-") if part]
    if len(parts) < 2 or parts[-1].isdigit():
        return False
    return any(part in _SCHEME_ALIAS_PARTS for part in parts)


def _derived_title_aliases(text: str) -> list[str]:
    cleaned = re.sub(r"\.pdf$", "", str(text or ""), flags=re.IGNORECASE)
    tokens = [token for token in re.findall(r"[a-z0-9]+", cleaned.lower()) if token]
    if not tokens:
        return []

    stopwords = {
        "a",
        "an",
        "the",
        "for",
        "of",
        "with",
        "and",
        "to",
        "from",
        "on",
        "in",
        "using",
        "toward",
        "towards",
        "based",
    }
    aliases = set()

    def add_alias_before(anchor_tokens: list[str], suffix: str, min_prefix: int = 2, max_prefix: int = 4):
        anchor_len = len(anchor_tokens)
        for index in range(len(tokens) - anchor_len + 1):
            if tokens[index : index + anchor_len] != anchor_tokens:
                continue
            prefix_tokens = [token for token in tokens[:index] if token not in stopwords]
            prefix_tokens = prefix_tokens[-max_prefix:]
            if len(prefix_tokens) < min_prefix:
                continue
            aliases.add(("".join(token[0] for token in prefix_tokens) + f"-{suffix}").lower())

    add_alias_before(["attribute", "based", "encryption"], "ABE")
    add_alias_before(["bilateral", "access", "control"], "BAC", min_prefix=1, max_prefix=3)
    return sorted(alias for alias in aliases if alias not in _GENERIC_PAPER_ALIASES)


def _paper_aliases(paper_record: dict) -> list[str]:
    aliases = set()
    hyphenated_pattern = r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b"
    standalone_pattern = r"\b[A-Z]{4,}\b"

    for value in (paper_record.get("title", ""), paper_record.get("source_name", "")):
        text = str(value or "")
        for match in re.findall(hyphenated_pattern, text):
            aliases.add(match.lower())
        for match in re.findall(standalone_pattern, text):
            aliases.add(match.lower())
        aliases.update(_derived_title_aliases(text))

    markdown_path = paper_record.get("markdown_path")
    if markdown_path:
        try:
            text = Path(markdown_path).read_text(encoding="utf-8")
            for match in re.findall(r"\((" + hyphenated_pattern[2:-2] + r")\)", text[:8000]):
                aliases.add(match.lower())
            for match in re.findall(hyphenated_pattern, text[:8000]):
                aliases.add(match.lower())
        except Exception:
            pass

    aliases.discard("")
    aliases = {alias for alias in aliases if alias not in _GENERIC_PAPER_ALIASES}
    return sorted(aliases)


def _resolve_query_aliases(query: str, paper_catalog: list[dict]) -> list[str]:
    query_text = str(query or "")
    normalized_query = query_text.lower()
    query_alias_tokens = set(_extract_query_alias_tokens(query_text))
    matched = []
    for paper in paper_catalog:
        aliases = _paper_aliases(paper)
        if any(alias and (alias in query_alias_tokens or re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_query)) for alias in aliases):
            paper_id = paper.get("paper_id", "")
            if paper_id and paper_id not in matched:
                matched.append(paper_id)
    return matched


def _recent_alternative_paper_ids(working_memory: dict, current_focus: str | None) -> list[str]:
    recent = [paper_id for paper_id in (working_memory.get("recent_papers", []) or []) if paper_id]
    return [paper_id for paper_id in recent if paper_id != current_focus]


def _previous_dialogue_paper_id(
    working_memory: dict,
    workspace_context: dict,
    paper_catalog: list[dict],
    current_dialogue_paper: str | None,
) -> str | None:
    recent = [paper_id for paper_id in (working_memory.get("recent_papers", []) or []) if paper_id]
    for paper_id in recent:
        if paper_id and paper_id != current_dialogue_paper:
            return paper_id

    anchor_paper = (
        current_dialogue_paper
        or working_memory.get("focus_paper_id")
        or workspace_context.get("selected_focus_paper_id")
        or workspace_context.get("focus_paper_id")
    )
    catalog_ids = [paper.get("paper_id", "") for paper in paper_catalog if paper.get("paper_id")]
    if anchor_paper in catalog_ids:
        index = catalog_ids.index(anchor_paper)
        if index > 0:
            return catalog_ids[index - 1]
    return None


def _merge_recent_papers(existing: list[str], new: list[str]) -> list[str]:
    return _unique_non_empty(list(new or []) + list(existing or []))[:6]


def _carry_forward_research_question(query: str, working_memory: dict) -> str:
    if not _is_short_follow_up_query(query):
        return str(query or "")
    prior_question = str(working_memory.get("current_research_question", "") or "").strip()
    return prior_question or str(query or "")


def _reference_strings_for_paper(paper_record: dict) -> list[str]:
    references = []
    references.extend(_paper_aliases(paper_record))
    for value in (paper_record.get("title", ""), paper_record.get("source_name", "")):
        text = str(value or "").strip()
        if not text:
            continue
        references.append(text)
        if text.lower().endswith(".pdf"):
            references.append(text[:-4])
    return _unique_non_empty([item for item in references if item])


def _paper_question_template(query: str, paper_id: str | None, paper_catalog: list[dict]) -> str:
    template = str(query or "").strip()
    if not template or not paper_id:
        return template

    paper_record = next((paper for paper in paper_catalog if paper.get("paper_id") == paper_id), {})
    if not paper_record:
        return template

    placeholder = "\u8fd9\u7bc7\u8bba\u6587" if re.search(r"[\u4e00-\u9fff]", template) else "this paper"
    references = sorted(_reference_strings_for_paper(paper_record), key=len, reverse=True)
    for reference in references:
        ref_text = str(reference or "").strip()
        if not ref_text:
            continue
        if re.fullmatch(r"[a-z0-9-]+", ref_text.lower()):
            template = re.sub(
                rf"(?<![a-z0-9]){re.escape(ref_text)}(?![a-z0-9])",
                placeholder,
                template,
                flags=re.IGNORECASE,
            )
        else:
            template = template.replace(ref_text, placeholder)
    return re.sub(r"\s+", " ", template).strip()


def _resolved_dialogue_paper_id(
    query: str,
    intent_type: str,
    referenced_documents: list[str],
    current_dialogue_paper: str | None,
) -> str | None:
    if not referenced_documents:
        return current_dialogue_paper
    if (
        intent_type == "cross_doc_comparison"
        and current_dialogue_paper
        and _refers_to_current_paper(query)
        and current_dialogue_paper in referenced_documents
    ):
        return current_dialogue_paper
    return referenced_documents[0]


def rewrite_query(state: State, llm, workspace_memory, default_workspace_id: str):
    last_message = state["messages"][-1]
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    conversation_summary = state.get("conversation_summary", "")
    paper_catalog = workspace_memory.list_papers(workspace_id)
    paper_lines = [
        f"- {paper.get('paper_id')}: {paper.get('title') or paper.get('source_name', paper.get('paper_id'))}"
        + (f" | aliases: {', '.join(_paper_aliases(paper)[:6])}" if _paper_aliases(paper) else "")
        for paper in paper_catalog
    ]
    paper_catalog_text = "\n".join(paper_lines) if paper_lines else "- No papers uploaded yet."
    working_memory = state.get("working_memory", {}) or {}
    current_focus_paper = working_memory.get("focus_paper_id") or workspace_context.get("focus_paper_id")
    known_aliases = {alias for paper in paper_catalog for alias in _paper_aliases(paper)}
    explicit_alias_tokens = [
        token
        for token in _extract_query_alias_tokens(last_message.content)
        if token in known_aliases or _looks_like_scheme_alias_token(token)
    ]
    alias_matches = _resolve_query_aliases(last_message.content, paper_catalog)
    working_memory_hit = workspace_memory.search_working_memory_facts(workspace_id, last_message.content)
    semantic_hits = workspace_memory.search_semantic_memory(workspace_id, last_message.content, limit=3)
    profile_hits = workspace_memory.search_paper_profiles(workspace_id, last_message.content, limit=3)
    interaction_hits = workspace_memory.search_episodic_memory(workspace_id, last_message.content, limit=3)
    current_dialogue_paper = _dialogue_paper_id(working_memory, workspace_context, interaction_hits)
    previous_dialogue_paper = _previous_dialogue_paper_id(
        working_memory,
        workspace_context,
        paper_catalog,
        current_dialogue_paper,
    )
    workspace_scope_query = _is_workspace_scope_query(last_message.content)
    workspace_inventory_query = _looks_like_workspace_inventory_query(last_message.content)
    follow_up_query = _carry_forward_research_question(last_message.content, working_memory)
    memory_hits = []
    if working_memory_hit:
        memory_hits.append({"source": "working_memory", "content": str(working_memory_hit)})
    for fact in semantic_hits:
        memory_hits.append(
            {
                "source": f"semantic:{fact.get('fact_id', '')}",
                "content": str({k: v for k, v in fact.items() if k != "_score"}),
            }
        )
    for profile in profile_hits:
        memory_hits.append(
            {
                "source": f"paper_profile:{profile.get('paper_id', '')}",
                "content": str({k: v for k, v in profile.items() if k != "_score"}),
            }
        )
    for interaction in interaction_hits:
        memory_hits.append(
            {
                "source": "interaction",
                "content": str({k: v for k, v in interaction.items() if k != "_score"}),
            }
        )

    context_section = (
        f"Workspace ID:\n{workspace_id}\n\n"
        f"Knowledge Scope:\n{workspace_context.get('knowledge_scope', 'workspace_documents')}\n\n"
        f"Current Dialogue Paper:\n{current_dialogue_paper or ''}\n\n"
        f"Previous Dialogue Paper:\n{previous_dialogue_paper or ''}\n\n"
        f"Focused Paper:\n{current_focus_paper or ''}\n\n"
        f"UI Selected Paper:\n{workspace_context.get('selected_focus_paper_id', '') or ''}\n\n"
        f"Conversation Context:\n{conversation_summary}\n\n"
        f"Working Memory:\n{working_memory}\n\n"
        f"Relevant Memory Hits:\n{_format_memory_hits(memory_hits)}\n\n"
        f"Available Workspace Papers:\n{paper_catalog_text}\n\n"
        f"Follow-up Query Skeleton:\n{follow_up_query}\n\n"
        f"User Query:\n{last_message.content}\n"
    )

    fast_path_focus = current_dialogue_paper or current_focus_paper
    if (
        fast_path_focus
        and not explicit_alias_tokens
        and not _is_comparison_question(last_message.content)
        and not _contains_any(last_message.content, ["literature review", "survey", "\u7efc\u8ff0"])
    ):
        response = _heuristic_intent_analysis(
            last_message.content,
            workspace_context,
            working_memory,
            current_dialogue_paper,
        )
    else:
        llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(
            IntentAnalysis,
            method="function_calling",
        )
        try:
            response = _normalize_intent_response(
                llm_with_structure.invoke(
                    [SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)]
                )
            )
        except Exception:
            response = None

        if response is None:
            response = _heuristic_intent_analysis(
                last_message.content,
                workspace_context,
                working_memory,
                current_dialogue_paper,
            )

    response.task_intent = _normalize_task_intent(
        response.task_intent,
        last_message.content,
        response.intent_type,
        workspace_inventory_query=workspace_inventory_query,
    )

    if (
        _is_comparison_question(last_message.content)
        and response.intent_type in {"general_retrieval", "single_doc_close_reading"}
    ):
        response.intent_type = "cross_doc_comparison"
        response.task_intent = _normalize_task_intent(
            response.task_intent,
            last_message.content,
            response.intent_type,
            workspace_inventory_query=workspace_inventory_query,
        )

    if _is_short_follow_up_query(last_message.content) and working_memory.get("current_research_question"):
        response.resolved_query = follow_up_query
        if not response.rewritten_questions or response.rewritten_questions == [last_message.content]:
            response.rewritten_questions = [follow_up_query]
        if response.intent_type == "general_retrieval":
            response.intent_type = workspace_context.get("intent_type", "single_doc_close_reading")

    referenced_documents = response.referenced_documents or []
    if _refers_to_previous_paper(last_message.content) and previous_dialogue_paper:
        referenced_documents = [previous_dialogue_paper]
        if response.intent_type == "general_retrieval":
            response.intent_type = "single_doc_close_reading"
    if alias_matches:
        referenced_documents = _unique_non_empty(alias_matches + referenced_documents)
        if response.intent_type in {"cross_doc_comparison", "literature_review"}:
            response.comparison_targets = _unique_non_empty(
                alias_matches + (response.comparison_targets or []) + referenced_documents
            )
        if response.intent_type == "general_retrieval" and not workspace_scope_query:
            response.intent_type = "single_doc_close_reading" if len(referenced_documents) == 1 else "cross_doc_comparison"
        if response.needs_clarification and referenced_documents:
            response.needs_clarification = False
            response.clarification_question = ""
        current_focus = current_dialogue_paper or current_focus_paper
        if (
            response.intent_type == "cross_doc_comparison"
            and current_focus
            and current_focus not in referenced_documents
            and (len(alias_matches) == 1 or _refers_to_current_paper(last_message.content))
        ):
            referenced_documents = [current_focus] + referenced_documents
            response.comparison_targets = _unique_non_empty([current_focus] + (response.comparison_targets or alias_matches))
        elif (
            response.intent_type == "cross_doc_comparison"
            and current_focus
            and current_focus in alias_matches
            and _refers_to_current_paper(last_message.content)
        ):
            alternatives = _recent_alternative_paper_ids(working_memory, current_focus)
            if alternatives:
                referenced_documents = _unique_non_empty(alternatives[:1] + referenced_documents)
                response.comparison_targets = _unique_non_empty(alternatives[:1] + (response.comparison_targets or referenced_documents))
            else:
                response.needs_clarification = True
                if re.search(r"[\u4e00-\u9fff]", str(last_message.content or "")):
                    response.clarification_question = (
                        f"你提到“它”和 `{alias_matches[0]}`，但当前线程的焦点论文也是 `{alias_matches[0]}`。请明确另一篇论文的名称。"
                    )
                else:
                    response.clarification_question = (
                        f"You referred to 'it' and `{alias_matches[0]}`, but the current focus paper is already `{alias_matches[0]}`. Please name the other paper explicitly."
                    )
    elif explicit_alias_tokens:
        referenced_documents = []
        response.comparison_targets = []
        response.needs_clarification = True
        missing_alias = explicit_alias_tokens[0]
        if re.search(r"[\u4e00-\u9fff]", str(last_message.content or "")):
            response.clarification_question = (
                f"我没有在当前工作区匹配到 `{missing_alias}` 这篇论文。请确认论文简称，或直接说论文标题。"
            )
        else:
            response.clarification_question = (
                f"I couldn't match `{missing_alias}` to a paper in the current workspace. Please confirm the alias or provide the paper title."
            )

    if (current_dialogue_paper or current_focus_paper) and not referenced_documents and not explicit_alias_tokens and not workspace_scope_query:
        referenced_documents = [current_dialogue_paper or current_focus_paper]

    if _looks_like_literature_review_query(last_message.content):
        response.intent_type = "literature_review"
        response.task_intent = _normalize_task_intent(
            response.task_intent,
            last_message.content,
            response.intent_type,
            workspace_inventory_query=workspace_inventory_query,
        )
        if not response.comparison_targets or workspace_scope_query:
            response.comparison_targets = [paper.get("paper_id", "") for paper in paper_catalog if paper.get("paper_id")]
        if workspace_scope_query:
            referenced_documents = []

    response.target_papers = _unique_non_empty(response.target_papers or response.comparison_targets or referenced_documents)
    artifact_kind = response.artifact_kind or _detect_artifact_kind(last_message.content, response.intent_type)
    retrieval_plan = _build_retrieval_plan(
        query=response.resolved_query,
        intent_type=response.intent_type,
        referenced_documents=referenced_documents,
        comparison_targets=response.comparison_targets or referenced_documents,
        workspace_context=workspace_context,
        paper_catalog=paper_catalog,
        task_intent=response.task_intent,
        proposed_plan=response.retrieval_plan or {},
    )
    if response.intent_type == "cross_doc_comparison" and len(_unique_non_empty(retrieval_plan.get("paper_ids", []))) < 2:
        response.needs_clarification = True
        if re.search(r"[\u4e00-\u9fff]", str(last_message.content or "")):
            response.clarification_question = "这是一个对比问题，但我目前只锁定到一篇论文。请明确说出另一篇论文的名称或简称。"
        else:
            response.clarification_question = (
                "This is a comparison question, but I could only resolve one paper so far. "
                "Please name the other paper explicitly."
            )
    rewritten_questions = response.rewritten_questions or [response.resolved_query]
    resolved_focus_paper_id = (
        referenced_documents[0]
        if referenced_documents
        else (None if explicit_alias_tokens else (current_dialogue_paper or current_focus_paper))
    )
    if workspace_scope_query and response.intent_type in {"general_retrieval", "literature_review"}:
        resolved_focus_paper_id = None
    resolved_dialogue_paper_id = _resolved_dialogue_paper_id(
        last_message.content,
        response.intent_type,
        referenced_documents,
        current_dialogue_paper,
    )
    template_paper_ids = _unique_non_empty(retrieval_plan.get("paper_ids", []) or referenced_documents)
    research_question_template = (
        _paper_question_template(response.resolved_query, template_paper_ids[0], paper_catalog)
        if response.intent_type in {"single_doc_close_reading", "general_retrieval"} and len(template_paper_ids) == 1
        else response.resolved_query
    )
    updated_memory = {
        **working_memory,
        "workspace_id": workspace_id,
        "focus_paper_id": resolved_focus_paper_id,
        "current_dialogue_paper_id": resolved_dialogue_paper_id,
        "current_research_question": follow_up_query if _is_short_follow_up_query(last_message.content) else research_question_template,
        "recent_papers": _merge_recent_papers(working_memory.get("recent_papers", []), referenced_documents),
        "open_questions": [response.clarification_question] if response.needs_clarification else [],
    }
    memory_context = _build_memory_context(updated_memory, semantic_hits + profile_hits, interaction_hits)
    workspace_memory.save_working_memory_snapshot(workspace_id, updated_memory)
    if updated_memory.get("focus_paper_id"):
        workspace_memory.save_semantic_fact(
            workspace_id,
            "confirmed_focus_paper",
            {
                "kind": "focus_paper",
                "paper_id": updated_memory.get("focus_paper_id"),
                "query": response.resolved_query,
            },
        )
    for paper in paper_catalog:
        if paper.get("paper_id") in referenced_documents:
            workspace_memory.save_semantic_fact(
                workspace_id,
                f"paper_aliases::{paper.get('paper_id')}",
                {
                    "kind": "paper_aliases",
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title", ""),
                    "aliases": _paper_aliases(paper),
                },
            )

    if response.needs_clarification:
        return {
            "questionIsClear": False,
            "clarification_question": response.clarification_question or "Please clarify your question.",
            "messages": [AIMessage(content=response.clarification_question or "Please clarify your question.")],
            "intent_type": response.intent_type,
            "task_intent": response.task_intent,
            "target_papers": response.target_papers,
            "answer_format": retrieval_plan.get("answer_format", response.answer_format),
            "intent_confidence": response.confidence,
            "workspace_context": {
                **workspace_context,
                "intent_type": response.intent_type,
                "task_intent": response.task_intent,
                "answer_format": retrieval_plan.get("answer_format", response.answer_format),
                "intent_confidence": response.confidence,
                "focus_paper_id": updated_memory.get("focus_paper_id"),
                "current_dialogue_paper_id": updated_memory.get("current_dialogue_paper_id"),
                "retrieval_plan": retrieval_plan,
            },
            "working_memory": updated_memory,
            "memory_context": memory_context,
            "memory_hits": memory_hits,
            "retrieval_plan": retrieval_plan,
            "artifact_kind": artifact_kind,
            "workspace_inventory_query": workspace_inventory_query,
        }

    delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
    return {
        "questionIsClear": True,
        "workspace_inventory_query": workspace_inventory_query,
        "messages": delete_all,
        "originalQuery": last_message.content,
        "rewrittenQuestions": rewritten_questions,
        "intent_type": response.intent_type,
        "task_intent": response.task_intent,
        "target_papers": response.target_papers,
        "referenced_documents": referenced_documents,
        "comparison_targets": response.comparison_targets or referenced_documents,
        "clarification_question": "",
        "retrieval_plan": retrieval_plan,
        "answer_format": retrieval_plan.get("answer_format", response.answer_format),
        "intent_confidence": response.confidence,
        "artifact_kind": artifact_kind,
        "artifact_payload": {},
        "workspace_context": {
            **workspace_context,
            "intent_type": response.intent_type,
            "task_intent": response.task_intent,
            "answer_format": retrieval_plan.get("answer_format", response.answer_format),
            "intent_confidence": response.confidence,
            "focus_paper_id": updated_memory.get("focus_paper_id"),
            "current_dialogue_paper_id": updated_memory.get("current_dialogue_paper_id"),
            "retrieval_plan": retrieval_plan,
        },
        "working_memory": updated_memory,
        "memory_context": memory_context,
        "memory_hits": memory_hits,
    }


def request_clarification(state: State):
    return {}


def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    workspace_context = state.get("workspace_context", {}) or {}
    sys_content = get_orchestrator_prompt()
    if workspace_context.get("focus_paper_id"):
        sys_content += f"\nCurrent focus paper: {workspace_context['focus_paper_id']}"
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    task_intent = state.get("task_intent") or workspace_context.get("task_intent") or retrieval_plan.get("task_intent", "")
    if task_intent:
        sys_content += f"\nFine-grained task intent: {task_intent}"
    if retrieval_plan.get("answer_format"):
        sys_content += f"\nPreferred answer format: {retrieval_plan['answer_format']}"
    if retrieval_plan.get("need_memory"):
        sys_content += "\nPrioritize workspace memory and previous reading context before searching paper text."
    if retrieval_plan.get("need_metadata_filter"):
        sys_content += "\nThis is a metadata-style request; use structured workspace tools before semantic retrieval."
    sys_msg = SystemMessage(content=sys_content)
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary
        else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        first_tool_name = "list_workspace_papers" if _looks_like_workspace_inventory_query(state["question"]) else "search_child_chunks"
        force_search = HumanMessage(
            content=f"YOU MUST CALL '{first_tool_name}' AS THE FIRST STEP TO ANSWER THIS QUESTION."
        )
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,
    }


def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()
    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data\n\n"
            + "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the workspace."
    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"RESPONSE LANGUAGE: {_response_language_instruction(state)}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke(
        [SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)]
    )
    return {"messages": [response]}


def should_compress_context(
    state: AgentState,
    *,
    base_token_threshold: int,
    token_growth_factor: float,
) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]
    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)
                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids
    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary
    max_allowed = base_token_threshold + int(current_token_summary * token_growth_factor)
    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)


def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT - {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke(
        [SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)]
    )
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))
        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    retrieved_chunks, retrieved_parent_chunks, retrieved_paper_ids, retrieved_sections = _extract_retrieval_trace_from_messages(
        state["messages"]
    )
    rerank_backend = _extract_rerank_backend_from_messages(state["messages"])
    return {
        "final_answer": answer,
        "agent_answers": [
            {
                "index": state["question_index"],
                "question": state["question"],
                "answer": answer,
                "rerank_backend": rerank_backend,
            }
        ],
        "retrieved_chunks": retrieved_chunks,
        "retrieved_parent_chunks": retrieved_parent_chunks,
        "retrieved_paper_ids": retrieved_paper_ids,
        "retrieved_sections": retrieved_sections,
        "rerank_backend": rerank_backend,
    }


def _contains_any(text: str, tokens: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in tokens)


TASK_INTENTS = {
    "single_paper_qa",
    "single_paper_summary",
    "method_explanation",
    "multi_paper_comparison",
    "literature_review",
    "citation_finding",
    "workspace_memory_qa",
    "metadata_query",
}


def _task_intent_for_coarse_intent(intent_type: str) -> str:
    if intent_type == "cross_doc_comparison":
        return "multi_paper_comparison"
    if intent_type == "literature_review":
        return "literature_review"
    return "single_paper_qa"


def _looks_like_single_paper_summary_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "summarize this paper",
            "summarize the paper",
            "paper summary",
            "structured summary",
            "contribution, method",
            "contributions, method",
            "related work",
            "\u603b\u7ed3\u8fd9\u7bc7",
            "\u6982\u62ec\u8fd9\u7bc7",
            "\u63d0\u70bc",
            "\u8d21\u732e\u3001\u65b9\u6cd5",
            "\u653e\u8fdb",
            "\u76f8\u5173\u5de5\u4f5c",
        ],
    )


def _looks_like_method_explanation_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "method",
            "pipeline",
            "mechanism",
            "module",
            "loss",
            "loss function",
            "training objective",
            "architecture",
            "algorithm",
            "formula",
            "equation",
            "how does",
            "how do",
            "why effective",
            "\u65b9\u6cd5",
            "\u673a\u5236",
            "\u6d41\u7a0b",
            "\u6a21\u5757",
            "\u635f\u5931\u51fd\u6570",
            "\u516c\u5f0f",
            "\u7b97\u6cd5",
            "\u8f93\u5165\u8f93\u51fa",
        ],
    )


def _looks_like_citation_finding_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "which paper supports",
            "support this claim",
            "supports this point",
            "find evidence",
            "evidence snippet",
            "can cite",
            "citation",
            "cite",
            "related work sentence",
            "papers about",
            "\u54ea\u7bc7\u8bba\u6587\u652f\u6301",
            "\u652f\u6301\u8fd9\u4e2a\u89c2\u70b9",
            "\u627e\u8bc1\u636e",
            "\u53ef\u4ee5\u5f15\u7528",
            "\u5f15\u7528",
            "\u8bc1\u636e\u7247\u6bb5",
        ],
    )


def _looks_like_workspace_memory_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "previously read",
            "read before",
            "last time",
            "previous conclusion",
            "my notes",
            "user notes",
            "marked as important",
            "workspace memory",
            "\u4e4b\u524d\u8bfb\u8fc7",
            "\u4e0a\u6b21",
            "\u4e4b\u524d\u603b\u7ed3",
            "\u6211\u7684\u7b14\u8bb0",
            "\u5386\u53f2",
            "\u8bb0\u5fc6",
            "\u6807\u8bb0\u4e3a\u91cd\u8981",
        ],
    )


def _looks_like_metadata_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "how many papers",
            "paper count",
            "count papers",
            "sort by year",
            "after 2024",
            "since 2024",
            "unread",
            "reading list",
            "tag",
            "tags",
            "export",
            "\u591a\u5c11\u7bc7",
            "\u51e0\u7bc7",
            "\u6309\u5e74\u4efd",
            "\u5e74\u4e4b\u540e",
            "\u8fd8\u6ca1\u8bfb",
            "\u9605\u8bfb\u6e05\u5355",
            "\u6253\u6807\u7b7e",
            "\u6807\u7b7e",
            "\u5bfc\u51fa",
        ],
    )


def _infer_task_intent(query: str, intent_type: str, workspace_inventory_query: bool = False) -> str:
    if workspace_inventory_query or _looks_like_metadata_query(query):
        return "metadata_query"
    if _looks_like_workspace_memory_query(query):
        return "workspace_memory_qa"
    if _looks_like_citation_finding_query(query):
        return "citation_finding"
    if intent_type == "literature_review" or _looks_like_literature_review_query(query):
        return "literature_review"
    if intent_type == "cross_doc_comparison" or _is_comparison_question(query):
        return "multi_paper_comparison"
    if _looks_like_single_paper_summary_query(query):
        return "single_paper_summary"
    if _looks_like_method_explanation_query(query):
        return "method_explanation"
    return _task_intent_for_coarse_intent(intent_type)


def _normalize_task_intent(task_intent: str | None, query: str, intent_type: str, workspace_inventory_query: bool = False) -> str:
    if task_intent in TASK_INTENTS:
        inferred = _infer_task_intent(query, intent_type, workspace_inventory_query)
        if task_intent == "single_paper_qa" and inferred != "single_paper_qa":
            return inferred
        if inferred in {"metadata_query", "workspace_memory_qa", "literature_review", "multi_paper_comparison"} and task_intent != inferred:
            return inferred
        return task_intent
    return _infer_task_intent(query, intent_type, workspace_inventory_query)


def _normalize_lookup_text(text: str) -> str:
    normalized = re.sub(r"[_*`#]", " ", str(text or "").lower())
    normalized = re.sub(r"\b[ivxlcdm]+\b", " ", normalized)
    normalized = re.sub(r"\b[a-z]\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _extract_algorithm_terms(text: str) -> list[str]:
    lowered = str(text or "").lower()
    canonical_terms = [
        "setup",
        "ekgen",
        "dkgen",
        "keygen",
        "enc",
        "encrypt",
        "dec",
        "decrypt",
        "match",
        "tag",
        "zkproof",
        "zkver",
        "proofgen",
        "verify",
        "trapdoor",
        "tokengen",
    ]
    matched = []
    for term in canonical_terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            matched.append(term)
    return matched


def _extract_query_profile(text: str, available_sections: list[str] | None = None) -> dict:
    lowered = (text or "").lower()
    normalized = _normalize_lookup_text(text)
    algorithm_terms = _extract_algorithm_terms(text)

    explicit_section_hints = []
    for phrase in [
        "offline phase",
        "online phase",
        "system procedure",
        "system model",
        "security assumptions",
        "security goals",
        "attacker model",
        "threat model",
        "security model",
        "main idea",
        "contribution",
        "experimental analysis",
        "experimental results",
        "performance analysis",
        "performance evaluation",
        "evaluation",
        "blockchain",
        "conclusion",
        "construction",
        "concrete construction",
        "detailed construction",
        "our construction",
        "preliminaries",
        "algorithm",
        "formula",
        "equation",
        "proof",
        "notation",
        "definition",
        "workflow",
        "decryption",
        "key generation",
        "system initialization",
        "break glass",
        "break-glass",
    ]:
        if phrase in normalized:
            explicit_section_hints.append(phrase)

    if available_sections:
        for section in available_sections:
            normalized_section = _normalize_lookup_text(section)
            if not normalized_section:
                continue
            tokens = [token for token in normalized_section.split() if len(token) > 2]
            if not tokens:
                continue
            phrase = " ".join(tokens)
            if phrase in normalized:
                explicit_section_hints.append(phrase)
                continue
            if len(tokens) >= 2 and all(token in normalized for token in tokens[:2]):
                explicit_section_hints.append(phrase)

    if "security model" in normalized:
        explicit_section_hints.extend(["threat model", "security model"])
    if "threat model" in normalized:
        explicit_section_hints.extend(["threat model", "security model"])
    if "experimental settings" in normalized or "experimental setup" in normalized:
        explicit_section_hints.extend(["experimental analysis", "experimental results", "performance analysis"])
    if any(term in normalized for term in ("algorithm", "formula", "equation", "proof", "construction")):
        explicit_section_hints.extend(
            ["concrete construction", "detailed construction", "our construction", "preliminaries", "notation"]
        )
    if any(
        term in lowered
        for term in [
            "\u7b97\u6cd5",
            "\u516c\u5f0f",
            "\u6570\u5b66",
            "\u6784\u9020",
            "\u8bc1\u660e",
            "\u4f2a\u4ee3\u7801",
            "\u63a8\u5bfc",
            "\u5b9a\u7406",
            "\u5f15\u7406",
        ]
    ):
        explicit_section_hints.extend(
            ["concrete construction", "detailed construction", "our construction", "preliminaries", "notation"]
        )

    return {
        "is_process": _is_process_question(lowered) or _contains_any(
            lowered,
            [
                "workflow",
                "procedure",
                "phase",
                "how does",
                "how do",
                "algorithm",
                "construction",
                "\u6b65\u9aa4",
                "\u6d41\u7a0b",
            ],
        ),
        "is_algorithmic": bool(algorithm_terms) or _contains_any(
            lowered,
            [
                "algorithm",
                "algorithms",
                "formula",
                "equation",
                "construction",
                "proof",
                "pseudocode",
                "notation",
                "derive",
                "derivation",
                "setup",
                "keygen",
                "encrypt",
                "decrypt",
                "match",
                "zkproof",
                "zkver",
                "\u7b97\u6cd5",
                "\u516c\u5f0f",
                "\u6570\u5b66",
                "\u6784\u9020",
                "\u8bc1\u660e",
                "\u4f2a\u4ee3\u7801",
                "\u63a8\u5bfc",
                "\u5b9a\u7406",
                "\u5f15\u7406",
            ],
        ),
        "is_security": _contains_any(
            lowered,
            [
                "security",
                "threat",
                "attacker",
                "adversar",
                "assumption",
                "confidential",
                "unforge",
                "privacy",
                "soundness",
                "zero-knowledge",
                "zkp",
                "policy hiding",
            ],
        ),
        "is_performance": _contains_any(
            lowered,
            [
                "performance",
                "experimental",
                "evaluation",
                "experiment",
                "runtime",
                "latency",
                "cost",
                "overhead",
                "setup",
                "hardware",
                "ram",
                "cpu",
                "environment",
                "benchmark",
            ],
        ),
        "wants_setup": _contains_any(
            lowered,
            ["setup", "environment", "hardware", "cpu", "gpu", "ram", "workstation", "server"],
        ),
        "is_system": _contains_any(
            lowered,
            [
                "system model",
                "architecture",
                "entity",
                "entities",
                "role",
                "edge node",
                "authority",
                "owner",
                "receiver",
                "user",
            ],
        ),
        "is_contribution": _contains_any(
            lowered,
            ["contribution", "main idea", "innovation", "novel", "propose", "proposed"],
        ),
        "algorithm_terms": algorithm_terms,
        "explicit_section_hints": _unique_non_empty(explicit_section_hints),
    }


def _looks_like_literature_review_query(text: str) -> bool:
    lowered = (text or "").lower()
    review_verbs = ["summarize", "synthesize", "review", "survey", "overview", "\u7efc\u8ff0", "\u603b\u7ed3"]
    multi_doc_markers = [
        "across the",
        "across papers",
        "workspace papers",
        "three papers",
        "multiple papers",
        "current workspace papers",
        "papers in the workspace",
        "\u5f53\u524d\u5de5\u4f5c\u533a",
        "\u8fd9\u4e9b\u8bba\u6587",
        "\u5de5\u4f5c\u533a\u8bba\u6587",
        "\u5de5\u4f5c\u533a\u91cc\u7684\u8bba\u6587",
        "\u6240\u6709\u8bba\u6587",
    ]
    return any(token in lowered for token in review_verbs) and any(token in lowered for token in multi_doc_markers)


def _is_workspace_scope_query(text: str) -> bool:
    lowered = (text or "").lower()
    if _contains_any(
        lowered,
        [
            "workspace papers",
            "current workspace papers",
            "papers in the workspace",
            "all papers",
            "multiple papers",
            "across papers",
            "\u5f53\u524d\u5de5\u4f5c\u533a",
            "\u8fd9\u4e9b\u8bba\u6587",
            "\u5de5\u4f5c\u533a\u8bba\u6587",
            "\u5de5\u4f5c\u533a\u91cc\u7684\u8bba\u6587",
            "\u6240\u6709\u8bba\u6587",
        ],
    ):
        return True
    return "workspace" in lowered and _contains_any(
        lowered,
        [
            "paper",
            "papers",
            "title",
            "titles",
            "loaded",
            "available",
            "current workspace",
        ],
    )


def _looks_like_workspace_inventory_query(text: str) -> bool:
    lowered = (text or "").lower()
    if not _is_workspace_scope_query(lowered):
        return False
    return _contains_any(
        lowered,
        [
            "what papers",
            "which papers",
            "how many papers",
            "paper count",
            "count papers",
            "list the papers",
            "available in the current workspace",
            "currently loaded in the workspace",
            "paper titles",
            "titles of the",
            "titles currently loaded",
            "available papers",
            "\u591a\u5c11\u7bc7",
            "\u51e0\u7bc7",
        ],
    )


def workspace_inventory_response(state: State, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    papers = workspace_memory.list_papers(workspace_id)
    query = str(state.get("originalQuery", "") or "")
    lowered = query.lower()
    wants_titles = _contains_any(lowered, ["title", "titles", "\u6807\u9898"])
    wants_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))

    if not papers and wants_chinese:
        answer = "当前工作区里还没有已注册的论文。"
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": [],
            "retrieved_sections": [],
            "verification_status": "not_applicable",
        }

    if not papers:
        answer = (
            "当前工作区里还没有已注册的论文。"
            if wants_chinese
            else "There are no papers registered in the current workspace yet."
        )
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": [],
            "retrieved_sections": [],
            "verification_status": "not_applicable",
        }

    paper_ids = [paper.get("paper_id", "") for paper in papers if paper.get("paper_id")]
    if wants_chinese:
        lines = ["当前工作区包含以下论文：", ""]
        for index, paper in enumerate(papers, start=1):
            label = paper.get("title") or paper.get("source_name") or paper.get("paper_id", "")
            if wants_titles:
                lines.append(f"{index}. {label}")
            else:
                lines.append(f"{index}. {paper.get('paper_id', '')}: {label}")
        answer = "\n".join(lines).strip()
        workspace_memory.record_interaction(
            workspace_id,
            {
                "intent_type": state.get("intent_type"),
                "task_intent": state.get("task_intent"),
                "query": query,
                "paper_ids": paper_ids,
                "retrieved_sections": [],
                "verification_status": "not_applicable",
                "artifact_kind": state.get("artifact_kind", ""),
            },
        )
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": paper_ids,
            "retrieved_sections": [],
            "verification_status": "not_applicable",
        }

    if wants_chinese:
        lines = ["当前工作区包含以下论文：", ""]
        for index, paper in enumerate(papers, start=1):
            label = paper.get("title") or paper.get("source_name") or paper.get("paper_id", "")
            if wants_titles:
                lines.append(f"{index}. {label}")
            else:
                lines.append(f"{index}. {paper.get('paper_id', '')}: {label}")
        answer = "\n".join(lines).strip()
    else:
        intro = "The current workspace includes these paper titles:" if wants_titles else "The current workspace includes these papers:"
        lines = [intro, ""]
        for index, paper in enumerate(papers, start=1):
            label = paper.get("title") or paper.get("source_name") or paper.get("paper_id", "")
            if wants_titles:
                lines.append(f"{index}. {label}")
            else:
                lines.append(f"{index}. {paper.get('paper_id', '')}: {label}")
        answer = "\n".join(lines).strip()

    workspace_memory.record_interaction(
        workspace_id,
        {
            "intent_type": state.get("intent_type"),
            "task_intent": state.get("task_intent"),
            "query": query,
            "paper_ids": paper_ids,
            "retrieved_sections": [],
            "verification_status": "not_applicable",
            "artifact_kind": state.get("artifact_kind", ""),
        },
    )
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "retrieved_chunks": [],
        "retrieved_parent_chunks": [],
        "retrieved_paper_ids": paper_ids,
        "retrieved_sections": [],
        "verification_status": "not_applicable",
    }


def _paper_display_label(paper: dict) -> str:
    return paper.get("title") or paper.get("source_name") or paper.get("paper_id", "")


def _available_metadata_fields(papers: list[dict]) -> list[str]:
    fields = set()
    for paper in papers:
        fields.update(str(key) for key in paper.keys())
    return sorted(fields)


def _metadata_tags(paper: dict) -> list[str]:
    tags = paper.get("tags", [])
    if isinstance(tags, str):
        return [item.strip() for item in re.split(r"[,;]", tags) if item.strip()]
    if isinstance(tags, list):
        return [str(item).strip() for item in tags if str(item).strip()]
    return []


def _metadata_read_status(paper: dict) -> str:
    if "read_status" in paper:
        return str(paper.get("read_status") or "").lower()
    if "read" in paper:
        return "read" if bool(paper.get("read")) else "unread"
    return ""


def _metadata_year(paper: dict) -> int:
    try:
        return int(paper.get("year") or 0)
    except Exception:
        return 0


def _looks_like_metadata_write_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return _contains_any(
        lowered,
        [
            "tag this",
            "add tag",
            "mark as read",
            "mark unread",
            "set read",
            "export",
            "\u6253\u6807\u7b7e",
            "\u6dfb\u52a0\u6807\u7b7e",
            "\u6807\u8bb0\u5df2\u8bfb",
            "\u6807\u8bb0\u672a\u8bfb",
            "\u5bfc\u51fa",
        ],
    )


def _metadata_query_result(query: str, papers: list[dict]) -> dict:
    lowered = str(query or "").lower()
    result_papers = list(papers)
    missing_fields = []

    tag_match = re.search(r"(?:tag|tags|标签)\s*[:=]?\s*([a-zA-Z0-9_\-\u4e00-\u9fff]+)", lowered)
    if tag_match:
        tag = tag_match.group(1).strip().lower()
        if not any("tags" in paper for paper in papers):
            missing_fields.append("tags")
        result_papers = [paper for paper in result_papers if tag in {item.lower() for item in _metadata_tags(paper)}]

    wants_unread = _contains_any(lowered, ["unread", "not read", "\u672a\u8bfb", "\u8fd8\u6ca1\u8bfb"])
    wants_read = _contains_any(lowered, ["read papers", "already read", "\u5df2\u8bfb"])
    if wants_unread or wants_read:
        if not any("read_status" in paper or "read" in paper for paper in papers):
            missing_fields.append("read_status/read")
        desired = "unread" if wants_unread else "read"
        result_papers = [paper for paper in result_papers if _metadata_read_status(paper) == desired]

    if _contains_any(lowered, ["sort by year", "按年份", "按年"]) or re.search(r"\b(after|since)\s+\d{4}\b", lowered):
        if not any("year" in paper for paper in papers):
            missing_fields.append("year")
        result_papers = sorted(result_papers, key=_metadata_year, reverse=True)
    elif _contains_any(lowered, ["updated", "recent", "\u6700\u8fd1"]):
        result_papers = sorted(result_papers, key=lambda paper: str(paper.get("updated_at", "")), reverse=True)

    year_filter = re.search(r"\b(?:after|since)\s+(\d{4})\b", lowered)
    if year_filter:
        threshold = int(year_filter.group(1))
        result_papers = [paper for paper in result_papers if _metadata_year(paper) >= threshold]

    wants_count = _contains_any(lowered, ["how many", "count", "\u591a\u5c11", "\u51e0\u7bc7"])
    wants_titles = _contains_any(lowered, ["title", "titles", "\u6807\u9898"])
    return {
        "kind": "metadata_query",
        "count": len(result_papers),
        "total_count": len(papers),
        "papers": result_papers,
        "wants_count": wants_count,
        "wants_titles": wants_titles,
        "missing_fields": _unique_non_empty(missing_fields),
        "available_fields": _available_metadata_fields(papers),
    }


def _render_metadata_answer(query: str, result: dict) -> str:
    if not result["papers"] and result.get("missing_fields"):
        return (
            "The current catalog does not contain the requested metadata field(s): "
            + ", ".join(result["missing_fields"])
            + f". Available fields: {', '.join(result['available_fields']) or 'none'}."
        )

    lines = []
    if result.get("wants_count"):
        lines.append(f"The current query matches {result['count']} paper(s) out of {result['total_count']} registered paper(s).")
    else:
        lines.append("Metadata query result:")
    if result.get("missing_fields"):
        lines.append(f"Missing requested metadata field(s): {', '.join(result['missing_fields'])}.")
    for index, paper in enumerate(result["papers"], start=1):
        label = _paper_display_label(paper)
        extras = []
        if paper.get("year"):
            extras.append(f"year={paper.get('year')}")
        tags = _metadata_tags(paper)
        if tags:
            extras.append(f"tags={', '.join(tags)}")
        read_status = _metadata_read_status(paper)
        if read_status:
            extras.append(f"read_status={read_status}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        if result.get("wants_titles"):
            lines.append(f"{index}. {label}{suffix}")
        else:
            lines.append(f"{index}. {paper.get('paper_id', '')}: {label}{suffix}")
    if not result["papers"] and not result.get("missing_fields"):
        lines.append("No catalog records matched this metadata query.")
    return "\n".join(lines).strip()


def metadata_query_response(state: State, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    query = str(state.get("originalQuery", "") or "")
    papers = workspace_memory.list_papers(workspace_id)
    paper_ids = [paper.get("paper_id", "") for paper in papers if paper.get("paper_id")]

    if _looks_like_metadata_write_query(query):
        answer = (
            "This metadata chain is read-only in the current version. "
            "It can answer catalog queries, but it will not modify tags, read status, or export files."
        )
    else:
        answer = _render_metadata_answer(query, _metadata_query_result(query, papers))

    workspace_memory.record_interaction(
        workspace_id,
        {
            "intent_type": state.get("intent_type"),
            "task_intent": state.get("task_intent", "metadata_query"),
            "query": query,
            "paper_ids": paper_ids,
            "retrieved_sections": [],
            "verification_status": "not_applicable",
            "artifact_kind": state.get("artifact_kind", ""),
        },
    )
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "retrieved_chunks": [],
        "retrieved_parent_chunks": [],
        "retrieved_paper_ids": paper_ids,
        "retrieved_sections": [],
        "verification_status": "not_applicable",
    }


def _memory_evidence_for_query(workspace_id: str, query: str, state: State, workspace_memory) -> list[dict]:
    evidence = []
    working_memory = state.get("working_memory", {}) or {}
    memory_context = state.get("memory_context", {}) or {}
    for item in state.get("memory_hits", []) or []:
        evidence.append({"source": item.get("source", "memory_hit"), "content": item.get("content", "")})
    if working_memory:
        evidence.append({"source": "working_memory", "content": str(working_memory)})
    if memory_context.get("session"):
        evidence.append({"source": "memory_context:session", "content": str(memory_context["session"])})
    for item in memory_context.get("semantic", []) or []:
        evidence.append({"source": f"memory_context:semantic:{item.get('fact_id', '')}", "content": str(item)})
    for item in memory_context.get("episodic", []) or []:
        evidence.append({"source": "memory_context:episodic", "content": str(item)})

    fact = workspace_memory.search_working_memory_facts(workspace_id, query)
    if fact:
        evidence.append({"source": "working_memory_search", "content": str(fact)})
    for item in workspace_memory.search_semantic_memory(workspace_id, query, limit=5):
        evidence.append({"source": f"semantic:{item.get('fact_id', '')}", "content": str({k: v for k, v in item.items() if k != "_score"})})
    for item in workspace_memory.search_paper_profiles(workspace_id, query, limit=5):
        evidence.append({"source": f"paper_profile:{item.get('paper_id', '')}", "content": str({k: v for k, v in item.items() if k != "_score"})})
    for item in workspace_memory.search_episodic_memory(workspace_id, query, limit=5):
        evidence.append({"source": "interaction_history", "content": str({k: v for k, v in item.items() if k != "_score"})})

    seen = set()
    deduped = []
    for item in evidence:
        key = (item.get("source", ""), item.get("content", ""))
        if key in seen or not str(item.get("content", "")).strip():
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:12]


def _fallback_memory_answer(query: str, evidence: list[dict]) -> str:
    if not evidence:
        return "I do not have enough relevant information in the current workspace memory to answer that."
    lines = ["I found relevant workspace memory:"]
    for item in evidence[:6]:
        lines.append(f"- [{item.get('source', 'memory')}] {str(item.get('content', ''))[:260]}")
    lines.append("\nSources: workspace memory")
    return "\n".join(lines)


def workspace_memory_qa(state: State, llm, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    query = str(state.get("originalQuery") or (state.get("rewrittenQuestions") or [""])[0])
    evidence = _memory_evidence_for_query(workspace_id, query, state, workspace_memory)
    verification_status = "pass" if evidence else "downgrade"
    if not evidence:
        answer = _fallback_memory_answer(query, evidence)
    else:
        prompt_content = (
            f"User question:\n{query}\n\n"
            f"Response language:\n{_response_language_instruction(state)}\n\n"
            f"Memory evidence:\n{_format_memory_hits(evidence)}"
        )
        try:
            response = llm.invoke([SystemMessage(content=get_workspace_memory_qa_prompt()), HumanMessage(content=prompt_content)])
            answer = getattr(response, "content", "") or ""
        except Exception:
            answer = ""
        if not answer.strip():
            answer = _fallback_memory_answer(query, evidence)

    workspace_memory.record_interaction(
        workspace_id,
        {
            "intent_type": state.get("intent_type"),
            "task_intent": state.get("task_intent", "workspace_memory_qa"),
            "query": query,
            "paper_ids": _unique_non_empty([str(item.get("paper_id", "")) for item in evidence]),
            "retrieved_sections": [],
            "verification_status": verification_status,
            "artifact_kind": state.get("artifact_kind", ""),
        },
    )
    memory_records = [
        {
            "source": item.get("source", "workspace_memory"),
            "content": item.get("content", ""),
            "content_type": "workspace_memory",
        }
        for item in evidence
    ]
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "retrieved_chunks": memory_records,
        "retrieved_parent_chunks": [],
        "retrieved_paper_ids": [],
        "retrieved_sections": [],
        "verification_status": verification_status,
    }


def _query_variants(query: str, query_profile: dict) -> list[str]:
    variants = [query]
    if query_profile.get("is_process"):
        variants.append(f"{query}\nmethod workflow process steps algorithm framework design")
    if query_profile.get("is_algorithmic"):
        variants.append(
            f"{query}\nconcrete construction detailed construction our construction preliminaries "
            "algorithm formula equation proof notation pseudocode"
        )
        variants.append(
            f"{query}\nsetup keygen ekgen dkgen encrypt enc decrypt dec match zkproof zkver detailed construction"
        )
        for term in query_profile.get("algorithm_terms", []):
            variants.append(term)
            variants.append(f"{query}\n{term}\nconcrete construction detailed construction algorithm")
    if query_profile.get("is_security"):
        variants.append(f"{query}\nsecurity model threat model attacker assumptions confidentiality unforgeability privacy")
    if query_profile.get("is_performance"):
        variants.append(f"{query}\nperformance evaluation experimental setup environment hardware runtime overhead cost")
    if query_profile.get("is_system"):
        variants.append(f"{query}\nsystem model architecture entities roles edge node authority owner receiver")
    if query_profile.get("is_contribution"):
        variants.append(f"{query}\ncontribution main idea proposed scheme novelty motivation")
    for hint in query_profile.get("explicit_section_hints", []):
        variants.append(f"{query}\nsection {hint}")
        variants.append(hint)
    return _unique_non_empty(variants)


def _section_matches_hints(section_name: str, hints: list[str]) -> bool:
    normalized_section = _normalize_lookup_text(section_name)
    return any(hint in normalized_section for hint in hints)


def _is_process_question(text: str) -> bool:
    return _contains_any(text, ["\u6b65\u9aa4", "\u6d41\u7a0b", "step", "process", "pipeline"])


def _is_comparison_question(text: str) -> bool:
    return _contains_any(
        text,
        [
            "\u76f8\u6bd4",
            "\u6bd4\u8f83",
            "\u5bf9\u6bd4",
            "\u533a\u522b",
            "\u5dee\u5f02",
            "\u4e0d\u540c",
            "\u5f02\u540c",
            "vs",
            "versus",
            "compared to",
            "compare",
            "difference",
            "differences",
            "different",
            "distinction",
        ],
    )


def _wants_citations(text: str) -> bool:
    return _contains_any(text, ["\u7ae0\u8282", "\u9875\u7801", "section", "page"])


def _section_priority(metadata: dict, query_profile: dict | None = None) -> tuple[int, int]:
    query_profile = query_profile or {}
    section_type = str(metadata.get("section_type", "")).lower()
    section_name = str(metadata.get("section", "")).lower()
    page_start = int(metadata.get("page_start", 9999) or 9999)

    normalized_section = _normalize_lookup_text(section_name)
    explicit_section_hints = query_profile.get("explicit_section_hints", [])
    if explicit_section_hints and any(hint in normalized_section for hint in explicit_section_hints):
        return (0, page_start)

    if query_profile.get("is_algorithmic"):
        if any(
            key in section_name
            for key in (
                "concrete construction",
                "detailed construction",
                "our construction",
                "construction",
                "algorithm",
                "scheme construction",
            )
        ):
            return (1, page_start)
        if any(
            key in section_name
            for key in (
                "system procedure",
                "system model",
                "workflow",
                "preliminaries",
                "notation",
                "definition",
                "correctness",
                "proof",
            )
        ):
            return (2, page_start)
        if section_type == "method":
            return (2, page_start)
        if any(key in section_name for key in ("contribution", "motivation", "introduction", "overview")):
            return (3, page_start)
        if any(key in section_name for key in ("performance", "experimental", "evaluation", "results", "conclusion")):
            return (4, page_start)
        return (5, page_start)

    if query_profile.get("is_process"):
        if section_type == "method" or any(
            key in section_name for key in ("method", "approach", "framework", "model", "scheme", "workflow", "design", "construction")
        ):
            return (1, page_start)
        if any(key in section_name for key in ("offline phase", "online phase", "system procedure", "decryption", "key generation")):
            return (1, page_start)
        if any(key in section_name for key in ("system model", "architecture", "overview")):
            return (2, page_start)
        if section_type == "introduction" or any(key in section_name for key in ("contribution", "motivation", "preliminaries")):
            return (3, page_start)
        if section_type == "experiment":
            return (4, page_start)
        return (5, page_start)

    if query_profile.get("is_security"):
        if any(key in section_name for key in ("security goals", "attacker model", "threat model", "security model", "security assumptions")):
            return (1, page_start)
        if any(key in section_name for key in ("security", "proof", "correctness", "privacy")):
            return (2, page_start)
        if query_profile.get("is_system") and "system model" in section_name:
            return (3, page_start)

    if query_profile.get("is_performance"):
        if any(key in section_name for key in ("performance", "experimental", "evaluation", "cost", "results", "analysis")):
            return (1, page_start)
        if "conclusion" in section_name:
            return (2, page_start)

    if query_profile.get("is_system"):
        if any(key in section_name for key in ("system model", "architecture", "system procedure", "workflow")):
            return (1, page_start)
        if any(key in section_name for key in ("threat model", "security assumptions")):
            return (2, page_start)

    if query_profile.get("is_contribution"):
        if any(key in section_name for key in ("contribution", "main idea", "motivation", "conclusion")):
            return (1, page_start)

    if section_type == "abstract":
        return (3, page_start)
    return (2, page_start)


def _detect_formula_signal(text: str) -> dict:
    source = text or ""
    lowered = source.lower()
    algorithm_markers = [
        "setup",
        "keygen",
        "encrypt",
        "decrypt",
        "atthid",
        "decryptout",
        "encryptout",
        "encrypton",
        "encryptoff",
        "keysymgen",
        "algorithm",
    ]
    formula_markers = [
        "bilinear",
        "lsss",
        "rlwe",
        "q-sdh",
        "_pp_",
        "_msk_",
        "_sk",
        "_pk",
        "_ct",
        "_lambda_",
        "_e_",
        "e(",
        "z _p",
        "g _t",
        "→",
        "lambda",
    ]
    damage_markers = ["�", "[′]", "[鈥", "_[", "]_[", "� _", "鈫", "脳", "锟", "談", "藛"]
    has_algorithm_markers = any(token in lowered for token in algorithm_markers)
    has_formula_markers = any(token in lowered for token in formula_markers)
    has_damage_markers = any(token in source for token in damage_markers)
    has_formula_gaps = any(
        token in lowered
        for token in [
            "formula-not-decoded",
            "<!-- formula-not-decoded -->",
            "equation-not-decoded",
            "<!-- equation-not-decoded -->",
        ]
    )
    return {
        "has_algorithm_markers": has_algorithm_markers,
        "has_formula_markers": has_formula_markers,
        "has_damage_markers": has_damage_markers,
        "has_formula_gaps": has_formula_gaps,
    }


_QUERY_TERM_STOPWORDS = {
    "about",
    "across",
    "algorithm",
    "algorithms",
    "answer",
    "asked",
    "asks",
    "current",
    "details",
    "does",
    "explain",
    "focused",
    "general",
    "give",
    "main",
    "major",
    "mentioned",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "please",
    "question",
    "report",
    "reported",
    "review",
    "scheme",
    "section",
    "sections",
    "selected",
    "score",
    "scores",
    "show",
    "shown",
    "state",
    "study",
    "summarize",
    "summary",
    "tell",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "work",
    "workspace",
}

_SHORT_QUERY_TERMS = {"abe", "cpu", "gpu", "zkp", "bleu"}

_FALLBACK_MARKERS = (
    "i could not find enough supporting passages",
    "not enough supporting passages",
    "insufficient evidence",
    "missing evidence",
    "current workspace",
    "selected paper",
)


def _salient_query_terms(query: str) -> list[str]:
    lowered = str(query or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered))
    terms = []
    for token in sorted(tokens):
        parts = [part for part in token.split("-") if part]
        if len(parts) > 1 and any(part in {"abe", "bac", "pbac"} for part in parts):
            continue
        if len(parts) > 1 and all(part.isdigit() or len(part) <= 4 for part in parts):
            continue
        if token in _QUERY_TERM_STOPWORDS:
            continue
        if len(token) < 4 and token not in _SHORT_QUERY_TERMS:
            continue
        if token.isdigit():
            continue
        terms.append(token)
    return terms


def _evidence_support_summary(query: str, evidence_items: list[dict]) -> dict:
    terms = _salient_query_terms(query)
    haystack = _normalize_lookup_text(
        " ".join(
            " ".join(
                [
                    str(item.get("section", "") or ""),
                    str(item.get("source", "") or ""),
                    str(item.get("content", "") or ""),
                ]
            )
            for item in evidence_items
        )
    )
    haystack_tokens = set(haystack.split())
    matched_terms = []
    for term in terms:
        normalized_term = _normalize_lookup_text(term)
        if not normalized_term:
            continue
        if " " in normalized_term:
            if re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", haystack):
                matched_terms.append(term)
        elif normalized_term in haystack_tokens:
            matched_terms.append(term)
    ratio = len(matched_terms) / len(terms) if terms else 0.0
    return {
        "terms": terms,
        "matched_terms": matched_terms,
        "support_ratio": ratio,
    }


def _is_fallback_answer(answer: str) -> bool:
    lowered = str(answer or "").lower()
    return any(marker in lowered for marker in _FALLBACK_MARKERS)


def _standard_fallback_answer(state: State) -> str:
    if state.get("intent_type") == "single_doc_close_reading" or (state.get("workspace_context", {}) or {}).get("focus_paper_id"):
        return "I could not find enough supporting passages in the selected paper to answer that question."
    return "I could not find enough supporting passages in the current workspace to answer that question."


def _fallback_close_reading_answer(original_query: str, paper_label: str, evidence_blocks: list[str]) -> str:
    preview = "\n\n".join(evidence_blocks[:3]).strip()
    if not preview:
        return "I could not generate a full answer, and the retrieved evidence was empty."
    return (
        f"I could not fully synthesize a final answer for this question: {original_query}\n\n"
        f"Here are the most relevant evidence blocks from {paper_label} that you can inspect directly:\n\n"
        f"{preview}\n\n"
        f"Sources\n{paper_label}"
    )


SUMMARY_SECTION_GROUPS = {
    "abstract": ["abstract", "\u6458\u8981"],
    "introduction": ["introduction", "background", "\u5f15\u8a00", "\u7eea\u8bba"],
    "method": ["method", "methodology", "construction", "framework", "algorithm", "system model", "\u65b9\u6cd5", "\u6784\u9020", "\u7b97\u6cd5"],
    "experiment_result": ["experiment", "experimental", "evaluation", "performance", "result", "analysis", "\u5b9e\u9a8c", "\u6027\u80fd", "\u7ed3\u679c"],
    "conclusion": ["conclusion", "discussion", "\u7ed3\u8bba"],
    "limitation": ["limitation", "future work", "threats", "\u5c40\u9650", "\u4e0d\u8db3", "\u672a\u6765\u5de5\u4f5c"],
}


def _summary_group_for_section(section_name: str) -> str | None:
    normalized = _normalize_lookup_text(section_name)
    for group, hints in SUMMARY_SECTION_GROUPS.items():
        if any(_normalize_lookup_text(hint) in normalized for hint in hints):
            return group
    return None


def _select_summary_evidence(parent_chunks: list[dict], per_group_limit: int = 3) -> dict[str, list[dict]]:
    grouped = {group: [] for group in SUMMARY_SECTION_GROUPS}
    fallback = []
    for item in parent_chunks:
        metadata = item.get("metadata", {}) or {}
        group = _summary_group_for_section(metadata.get("section", ""))
        record = _record_from_parent_chunk(item)
        if group and len(grouped[group]) < per_group_limit:
            grouped[group].append(record)
        elif len(fallback) < per_group_limit:
            fallback.append(record)
    if not grouped["introduction"] and fallback:
        grouped["introduction"] = fallback[:per_group_limit]
    return grouped


def _format_summary_evidence_map(grouped: dict[str, list[dict]]) -> str:
    blocks = []
    for group, records in grouped.items():
        if not records:
            blocks.append(f"## {group}\nNo parsed evidence selected for this section.")
            continue
        snippets = []
        for item in records:
            snippets.append(
                "\n".join(
                    [
                        f"Section: {item.get('section') or 'Unknown'}",
                        f"Pages: {item.get('pages') or '?'}",
                        f"Source: {item.get('source') or item.get('paper_id')}",
                        f"Content: {item.get('content', '')[:1200]}",
                    ]
                )
            )
        blocks.append(f"## {group}\n" + "\n\n---\n\n".join(snippets))
    return "\n\n".join(blocks)


def _fallback_single_paper_summary(query: str, paper_label: str, grouped: dict[str, list[dict]]) -> str:
    labels = {
        "abstract": "Contributions",
        "introduction": "Background",
        "method": "Method",
        "experiment_result": "Experiments / Results",
        "conclusion": "Conclusion",
        "limitation": "Limitations",
    }
    lines = [f"Structured summary for {paper_label}", ""]
    for group, heading in labels.items():
        records = grouped.get(group, [])
        lines.append(f"## {heading}")
        if records:
            lines.append(records[0].get("content", "")[:360].strip())
        else:
            lines.append("The current parsed content does not cover this section.")
        lines.append("")
    lines.extend(["## Related work sentence", "The current parsed evidence is insufficient to write a precise related-work sentence without overclaiming.", "", "Sources", paper_label])
    return "\n".join(lines).strip()


def single_paper_summary(state: State, llm, parent_store, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    working_memory = state.get("working_memory", {}) or {}
    paper_id = (
        (retrieval_plan.get("paper_ids") or [None])[0]
        or (state.get("referenced_documents") or [None])[0]
        or working_memory.get("current_dialogue_paper_id")
        or workspace_context.get("focus_paper_id")
    )
    if not paper_id:
        answer = "I could not determine which paper to summarize in the current workspace."
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": [],
            "retrieved_sections": [],
        }

    original_query = state.get("originalQuery", "")
    paper_catalog = workspace_memory.list_papers(workspace_id)
    paper_record = next((paper for paper in paper_catalog if paper.get("paper_id") == paper_id), {})
    paper_label = paper_record.get("source_name") or paper_record.get("title") or paper_id
    try:
        parent_chunks = parent_store.load_paper(workspace_id, paper_id)
    except Exception:
        parent_chunks = []

    if not parent_chunks:
        answer = f"I could not find parsed parent chunks for `{paper_label}`, so I cannot summarize it reliably."
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": [paper_id],
            "retrieved_sections": [],
        }

    grouped = _select_summary_evidence(parent_chunks)
    evidence_map = _format_summary_evidence_map(grouped)
    prompt_content = (
        f"User request:\n{original_query}\n\n"
        f"Focused paper:\n{paper_label} ({paper_id})\n\n"
        f"Response language:\n{_response_language_instruction(state)}\n\n"
        f"Section evidence map:\n{evidence_map}"
    )
    try:
        response = llm.invoke([SystemMessage(content=get_single_paper_summary_prompt()), HumanMessage(content=prompt_content)])
        final_answer = getattr(response, "content", "") or ""
    except Exception:
        final_answer = ""
    if not final_answer.strip():
        final_answer = _fallback_single_paper_summary(original_query, paper_label, grouped)

    parent_records = []
    for records in grouped.values():
        parent_records.extend(records)
    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "retrieved_chunks": [],
        "retrieved_parent_chunks": parent_records,
        "retrieved_paper_ids": [paper_id],
        "retrieved_sections": _unique_non_empty([item.get("section", "") for item in parent_records]),
    }


def close_reading(state: State, llm, evidence_retriever, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    working_memory = state.get("working_memory", {}) or {}
    paper_id = (
        (retrieval_plan.get("paper_ids") or [None])[0]
        or (state.get("referenced_documents") or [None])[0]
        or working_memory.get("current_dialogue_paper_id")
        or workspace_context.get("focus_paper_id")
    )

    if not paper_id:
        return {
            "messages": [AIMessage(content="I could not determine which paper to read closely in the current workspace.")],
            "final_answer": "I could not determine which paper to read closely in the current workspace.",
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": [],
            "retrieved_sections": [],
        }

    query = (state.get("rewrittenQuestions") or [state.get("originalQuery", "")])[0]
    original_query = state.get("originalQuery", "")
    lowered_query = f"{original_query}\n{query}".lower()
    paper_catalog = workspace_memory.list_papers(workspace_id)
    paper_record = next((paper for paper in paper_catalog if paper.get("paper_id") == paper_id), {})
    query_profile = _extract_query_profile(lowered_query, paper_record.get("sections", []))
    plan_profile = _retrieval_profile_from_plan(lowered_query, retrieval_plan)
    query_profile = {
        **query_profile,
        "explicit_section_hints": _unique_non_empty(
            query_profile.get("explicit_section_hints", []) + plan_profile.get("explicit_section_hints", [])
        ),
        "content_type_hints": _unique_non_empty(plan_profile.get("content_type_hints", [])),
    }
    task_intent = state.get("task_intent") or retrieval_plan.get("task_intent", "single_paper_qa")
    answer_format = state.get("answer_format") or retrieval_plan.get("answer_format", "short_answer")
    is_process_question = query_profile.get("is_process", False)
    is_algorithm_question = query_profile.get("is_algorithmic", False)
    is_comparison_question = _is_comparison_question(lowered_query)
    wants_citations = _wants_citations(lowered_query)
    query_variants = _query_variants(query, query_profile)
    retrieval_result = evidence_retriever.search(
        query,
        workspace_id=workspace_id,
        paper_id=paper_id,
        limit=int(retrieval_plan.get("per_paper_limit") or (6 if is_process_question or is_algorithm_question else 4)),
        query_profile=query_profile,
        query_variants=query_variants,
    )
    child_hits = retrieval_result["child_hits"]
    parent_chunks = retrieval_result["parent_chunks"]
    rerank_backend = retrieval_result["rerank_backend"]

    if not child_hits:
        answer = _standard_fallback_answer({**state, "intent_type": "single_doc_close_reading"})
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [],
            "retrieved_parent_chunks": [],
            "retrieved_paper_ids": [paper_id],
            "retrieved_sections": [],
            "rerank_backend": rerank_backend,
        }

    parent_chunk_records = [_record_from_parent_chunk(item) for item in parent_chunks]
    support_summary = _evidence_support_summary(original_query or query, parent_chunk_records)
    if support_summary["terms"] and not support_summary["matched_terms"]:
        answer = _standard_fallback_answer({**state, "intent_type": "single_doc_close_reading"})
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "retrieved_chunks": [_record_from_doc(doc) for doc in child_hits],
            "retrieved_parent_chunks": parent_chunk_records,
            "retrieved_paper_ids": [paper_id],
            "retrieved_sections": _unique_non_empty(
                [doc.metadata.get("section", "") for doc in child_hits]
                + [item.get("metadata", {}).get("section", "") for item in parent_chunks]
            ),
            "rerank_backend": rerank_backend,
        }

    evidence_limit = 6 if is_process_question or is_algorithm_question else 3
    evidence_blocks = []
    for item in parent_chunks[:evidence_limit]:
        metadata = item.get("metadata", {})
        evidence_blocks.append(
            "\n".join(
                [
                    f"Section: {metadata.get('section', 'Unknown')}",
                    f"Pages: {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}",
                    f"Source: {metadata.get('source', paper_id)}",
                    f"Content: {item.get('content', '').strip()}",
                ]
            )
        )

    if not evidence_blocks:
        for doc in child_hits:
            evidence_blocks.append(
                "\n".join(
                    [
                        f"Section: {doc.metadata.get('section', 'Unknown')}",
                        f"Pages: {doc.metadata.get('page_start', '?')}-{doc.metadata.get('page_end', '?')}",
                        f"Source: {doc.metadata.get('source', paper_id)}",
                        f"Content: {doc.page_content.strip()}",
                    ]
                )
            )

    paper_label = paper_record.get("source_name") or paper_record.get("title") or paper_id
    evidence_signal = _detect_formula_signal("\n\n".join(evidence_blocks))
    extra_instructions = []
    if is_comparison_question:
        extra_instructions.append(
            "This is a comparison question. Keep the comparison target exactly as named by the user. "
            "Do not introduce other schemes or side comparisons unless the paper excerpt explicitly says they are part of that exact comparison."
        )
        extra_instructions.append(
            "For each claimed improvement, state the evidence citation immediately after the claim using section name and page range."
        )
    if is_process_question:
        extra_instructions.append(
            "This is a process question. Prefer evidence from method, framework, workflow, construction, or system-model sections over generic contribution summaries."
        )
        extra_instructions.append(
            "Use this structure: (1) explicitly stated workflow elements from the paper, (2) cautiously reconstructed step-by-step flow, (3) missing algorithmic or procedural details."
        )
        extra_instructions.append(
            "If the paper does not present a formal workflow section, say so explicitly before giving the cautious reconstruction."
        )
    if is_algorithm_question:
        extra_instructions.append(
            "This is an algorithm-or-formula question. Prioritize concrete construction, detailed construction, preliminaries, notation, proof, and system-model sections over introduction, conclusion, or experimental results."
        )
        extra_instructions.append(
            "If algorithm headings such as Setup, KeyGen, Enc, Match, ZKProof, or ZKVer appear in the evidence, summarize those concrete steps before saying details are missing."
        )
        extra_instructions.append(
            "If formulas are partially damaged by PDF parsing, distinguish between recoverable algorithm structure and unreliable symbol-level details."
        )
    if wants_citations:
        extra_instructions.append(
            "The user explicitly asked for citations. Include section names and page ranges inline for every major point."
        )
    if task_intent == "single_paper_summary":
        extra_instructions.append(
            "This is a single-paper summarization task. Cover contributions, method, experiments/results, conclusion, and limitations when the evidence supports them."
        )
        extra_instructions.append(
            "If one of those expected sections is missing from the retrieved evidence, state that the current excerpts do not cover it instead of inventing it."
        )
    if task_intent == "method_explanation":
        extra_instructions.append(
            "This is a method explanation task. Reconstruct the mechanism chain: inputs, modules or phases, formulas/objectives if available, outputs, and why the parts matter."
        )
    if task_intent == "citation_finding" or answer_format == "evidence_list":
        extra_instructions.append(
            "This is an evidence-finding task. Prefer a compact evidence list with paper/source, claim, section/pages, snippet summary, and why it is relevant."
        )
    if evidence_signal["has_algorithm_markers"] or evidence_signal["has_formula_markers"]:
        extra_instructions.append(
            "The retrieved evidence contains algorithm-name or formula-like content. Do not say the paper lacks formulas or algorithm details entirely."
        )
    if evidence_signal["has_formula_gaps"]:
        extra_instructions.append(
            "The evidence contains explicit formula-decoding gaps. Do not restate exact formulas, exponents, or pairing equations unless the symbols are clearly intact in the excerpt."
        )
        extra_instructions.append(
            "For affected passages, describe the algorithmic role of the formula at a high level and clearly say the exact mathematical expression is incomplete in the parsed text."
        )
    if evidence_signal["has_damage_markers"]:
        extra_instructions.append(
            "Some mathematical notation in the evidence appears degraded by PDF parsing. If relevant, say the formulas are partially available but not fully reliable in the parsed text."
        )
        extra_instructions.append(
            "When discussing exact mathematical construction, explicitly recommend checking the original PDF for formula-level verification."
        )

    prompt_content = (
        f"User question:\n{original_query}\n\n"
        f"Resolved query:\n{query}\n\n"
        f"Focused paper:\n{paper_label} ({paper_id})\n\n"
        f"Task intent:\n{task_intent}\n\n"
        f"Preferred answer format:\n{answer_format}\n\n"
        f"Response language:\n{_response_language_instruction(state)}\n\n"
        "Evidence diagnostics:\n"
        f"- Algorithm-name markers detected: {evidence_signal['has_algorithm_markers']}\n"
        f"- Formula-like markers detected: {evidence_signal['has_formula_markers']}\n"
        f"- Parsing damage markers detected: {evidence_signal['has_damage_markers']}\n"
        f"- Formula gap markers detected: {evidence_signal['has_formula_gaps']}\n\n"
        f"Task-specific instructions:\n" + ("\n".join(f"- {item}" for item in extra_instructions) if extra_instructions else "- No extra instructions.") + "\n\n"
        f"Evidence:\n\n" + "\n\n---\n\n".join(evidence_blocks)
    )
    try:
        response = llm.invoke([SystemMessage(content=get_close_reading_prompt()), HumanMessage(content=prompt_content)])
        final_content = getattr(response, "content", "") if response is not None else ""
    except Exception:
        final_content = ""

    if not str(final_content).strip():
        final_content = _fallback_close_reading_answer(original_query, paper_label, evidence_blocks)

    parent_records = [_record_from_parent_chunk(item) for item in parent_chunks]
    artifact_payload, final_content = _apply_artifact_kind(state, final_content, evidence=parent_records)

    return {
        "messages": [AIMessage(content=final_content)],
        "final_answer": final_content,
        "artifact_payload": artifact_payload,
        "retrieved_chunks": [_record_from_doc(doc) for doc in child_hits],
        "retrieved_parent_chunks": parent_records,
        "retrieved_paper_ids": [paper_id],
        "retrieved_sections": _unique_non_empty(
            [doc.metadata.get("section", "") for doc in child_hits]
            + [item.get("metadata", {}).get("section", "") for item in parent_chunks]
        ),
        "rerank_backend": rerank_backend,
    }


def _target_papers(state: State, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    targets = retrieval_plan.get("paper_ids") or state.get("comparison_targets") or state.get("referenced_documents") or []
    if targets:
        return workspace_id, targets

    catalog = workspace_memory.list_papers(workspace_id)
    return workspace_id, [paper["paper_id"] for paper in catalog[:5]]


def _profile_trace_from_parent_ids(workspace_id, paper_id, source_parent_ids, parent_store):
    trace = []
    for parent_id in source_parent_ids:
        try:
            trace.append(parent_store.load_content(workspace_id, paper_id, parent_id))
        except Exception:
            continue
    return trace


def _dedupe_parent_chunk_records(items: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in items:
        parent_id = item.get("parent_id", "")
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        unique.append(item)
    return unique


def _interleave_parent_chunk_groups(groups: list[list[dict]], per_group_limit: int = 4) -> list[dict]:
    trimmed_groups = [_dedupe_parent_chunk_records(group)[:per_group_limit] for group in groups if group]
    merged = []
    index = 0
    while True:
        appended = False
        for group in trimmed_groups:
            if index < len(group):
                merged.append(group[index])
                appended = True
        if not appended:
            break
        index += 1
    return _dedupe_parent_chunk_records(merged)


def _profile_parent_chunk_priority(item: dict, query_text: str = "") -> tuple[int, int]:
    metadata = item.get("metadata", {}) or {}
    section_name = str(metadata.get("section", "")).lower()
    page_start = int(metadata.get("page_start", 9999) or 9999)
    query_profile = _extract_query_profile(query_text, [metadata.get("section", "")]) if query_text else {}
    explicit_section_hints = query_profile.get("explicit_section_hints", [])
    normalized_section = _normalize_lookup_text(section_name)

    if explicit_section_hints and any(hint in normalized_section for hint in explicit_section_hints):
        return (0, page_start)

    if query_profile.get("wants_setup") and any(
        token in section_name for token in ("experimental results", "experimental analysis")
    ):
        return (1, page_start)
    if query_profile.get("wants_setup") and any(
        token in section_name for token in ("performance analysis", "performance evaluation")
    ):
        return (2, page_start)
    if query_profile.get("is_performance") and any(
        token in section_name for token in ("experimental results", "experimental analysis")
    ):
        return (1, page_start)
    if query_profile.get("is_performance") and any(
        token in section_name for token in ("performance analysis", "performance evaluation")
    ):
        return (2, page_start)
    if query_profile.get("is_performance") and any(
        token in section_name for token in ("storage cost", "computation cost", "theoretical analysis", "correctness analysis")
    ):
        return (3, page_start)
    if query_profile.get("is_security") and any(
        token in section_name for token in ("security goals", "attacker model", "threat model", "security model", "security assumptions", "privacy")
    ):
        return (1, page_start)
    if query_profile.get("is_algorithmic") and any(
        token in section_name
        for token in (
            "concrete construction",
            "detailed construction",
            "our construction",
            "construction",
            "algorithm",
            "preliminaries",
            "notation",
            "proof",
            "correctness",
        )
    ):
        return (1, page_start)
    if query_profile.get("is_process") and any(
        token in section_name for token in ("offline phase", "online phase", "system procedure", "workflow", "construction", "decryption", "key generation")
    ):
        return (1, page_start)
    if query_profile.get("is_system") and any(
        token in section_name for token in ("system model", "architecture", "system procedure")
    ):
        return (1, page_start)
    if query_profile.get("is_contribution") and any(
        token in section_name for token in ("contribution", "main idea", "conclusion")
    ):
        return (1, page_start)

    if query_text:
        if query_profile.get("is_performance"):
            preferred_groups = [
                ("experimental results", "experimental analysis", "performance analysis", "performance evaluation"),
                ("performance", "experimental", "evaluation", "results"),
                ("system model", "architecture"),
                ("security model", "threat model", "security assumptions"),
                ("contribution", "main idea", "conclusion", "abstract"),
                ("construction", "scheme", "workflow", "cost"),
            ]
        elif query_profile.get("is_security"):
            preferred_groups = [
                ("security goals", "attacker model", "threat model", "security model", "security assumptions", "privacy"),
                ("security", "proof", "correctness"),
                ("system model", "architecture"),
                ("construction", "scheme", "workflow"),
                ("contribution", "main idea", "conclusion", "abstract"),
            ]
        elif query_profile.get("is_algorithmic"):
            preferred_groups = [
                ("concrete construction", "detailed construction", "our construction", "construction", "algorithm"),
                ("preliminaries", "notation", "definition", "system procedure", "workflow", "system model"),
                ("proof", "correctness", "security assumptions"),
                ("contribution", "main idea", "conclusion", "abstract"),
                ("performance", "experimental", "evaluation", "cost"),
            ]
        elif query_profile.get("is_process"):
            preferred_groups = [
                ("offline phase", "online phase", "system procedure", "workflow", "construction", "detailed construction"),
                ("system model", "architecture"),
                ("security model", "threat model", "security assumptions"),
                ("contribution", "main idea", "conclusion", "abstract"),
                ("performance", "experimental", "evaluation", "cost"),
            ]
        elif query_profile.get("is_system"):
            preferred_groups = [
                ("system model", "architecture", "system procedure"),
                ("threat model", "security assumptions", "security model"),
                ("construction", "scheme", "workflow"),
                ("contribution", "main idea", "conclusion", "abstract"),
                ("performance", "experimental", "evaluation", "cost"),
            ]
        elif query_profile.get("is_contribution"):
            preferred_groups = [
                ("contribution", "main idea", "conclusion", "abstract"),
                ("system model", "threat model", "security model", "security assumptions"),
                ("construction", "scheme", "detailed construction", "system procedure", "workflow"),
                ("performance", "experimental", "evaluation", "cost"),
                ("security", "proof", "correctness"),
            ]
        else:
            preferred_groups = [
                ("contribution", "main idea", "conclusion", "abstract"),
                ("system model", "threat model", "security model", "security assumptions"),
                ("construction", "scheme", "detailed construction", "system procedure", "workflow"),
                ("performance", "experimental", "evaluation", "cost"),
                ("security", "proof", "correctness"),
            ]
    else:
        preferred_groups = [
            ("contribution", "main idea", "conclusion", "abstract"),
            ("system model", "threat model", "security model", "security assumptions"),
            ("construction", "scheme", "detailed construction", "system procedure", "workflow"),
            ("performance", "experimental", "evaluation", "cost"),
            ("security", "proof", "correctness"),
        ]
    for priority, group in enumerate(preferred_groups):
        if any(token in section_name for token in group):
            return (priority + 2, page_start)
    return (len(preferred_groups), page_start)


def _select_supporting_parent_chunks(parent_chunks: list[dict], query_text: str, limit: int = 6) -> list[dict]:
    if not parent_chunks:
        return []
    query_profile = _extract_query_profile(query_text)
    hinted = []
    if query_profile.get("explicit_section_hints"):
        hinted = [
            item
            for item in parent_chunks
            if _section_matches_hints(item.get("metadata", {}).get("section", ""), query_profile["explicit_section_hints"])
        ]
    ranked = sorted(parent_chunks, key=lambda item: _profile_parent_chunk_priority(item, query_text))
    return _dedupe_parent_chunk_records(hinted + ranked)[:limit]


def _support_selection_query(state: State) -> str:
    original_query = str(state.get("originalQuery", "") or "").strip()
    if original_query:
        return original_query
    rewritten_questions = [str(item or "").strip() for item in (state.get("rewrittenQuestions") or []) if str(item or "").strip()]
    return rewritten_questions[0] if rewritten_questions else ""


def _response_language_instruction(state) -> str:
    original_query = str(state.get("originalQuery", "") or "").strip()
    if re.search(r"[\u4e00-\u9fff]", original_query):
        return "Respond entirely in Simplified Chinese."
    return "Respond in the same language as the user's question."


def _policy_from_state(state: State, default_workspace_id: str) -> RetrieverPolicy:
    workspace_context = state.get("workspace_context", {}) or {}
    return RetrieverPolicy.from_context(
        workspace_context,
        default_workspace_id=default_workspace_id,
        default_knowledge_scope=workspace_context.get("knowledge_scope", "workspace_documents"),
    )


def _retrieval_profile_from_plan(query: str, retrieval_plan: dict) -> dict:
    base_profile = _extract_query_profile(query)
    section_hints = retrieval_plan.get("section_hints", []) or []
    retrieval_scope = retrieval_plan.get("retrieval_scope", []) or []
    content_type_hints = retrieval_plan.get("content_type_hints", []) or []
    return {
        **base_profile,
        "explicit_section_hints": _unique_non_empty(base_profile.get("explicit_section_hints", []) + section_hints + retrieval_scope),
        "content_type_hints": _unique_non_empty(content_type_hints),
    }


def _retrieve_support_groups(
    state: State,
    evidence_retriever,
    workspace_memory,
    default_workspace_id: str,
    support_query: str,
) -> tuple[str, list[dict], list[dict], list[dict], str]:
    policy = _policy_from_state(state, default_workspace_id)
    workspace_id = policy.workspace_id
    paper_ids = policy.effective_paper_ids()
    if not paper_ids and policy.mode in {"literature_review", "per_paper_compare"}:
        paper_ids = [paper.get("paper_id", "") for paper in workspace_memory.list_papers(workspace_id) if paper.get("paper_id")]
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    query_profile = _retrieval_profile_from_plan(support_query, retrieval_plan)
    all_records = []
    support_groups = []
    rerank_backends = []
    for paper_id in paper_ids:
        result = evidence_retriever.search(
            support_query,
            workspace_id=workspace_id,
            paper_id=paper_id or None,
            limit=policy.per_paper_limit,
            query_profile=query_profile,
            query_variants=_query_variants(support_query, query_profile),
        )
        support_groups.append(result.get("parent_chunks", []) or [])
        all_records.extend([_record_from_parent_chunk(item) for item in result.get("parent_chunks", []) or []])
        rerank_backends.append(result.get("rerank_backend", ""))

    support_parent_chunks = _interleave_parent_chunk_groups(support_groups, per_group_limit=policy.per_paper_limit)
    merged_records = _dedupe_parent_chunk_records(all_records)[: policy.global_limit]
    backend = next((item for item in rerank_backends if item), state.get("rerank_backend", ""))
    return workspace_id, paper_ids, support_parent_chunks, merged_records, backend


def _format_notes_artifact(query: str, answer: str, evidence: list[dict]) -> tuple[dict, str]:
    grouped: dict[str, list[str]] = {}
    for item in evidence[:6]:
        section = item.get("section", "Unknown")
        grouped.setdefault(section, []).append(item.get("content", "")[:220].strip())
    payload = {
        "kind": "export_notes",
        "query": query,
        "sections": [{"section": section, "notes": notes} for section, notes in grouped.items()],
    }
    lines = ["# Research Notes", "", f"Question: {query}", ""]
    for section, notes in grouped.items():
        lines.append(f"## {section}")
        for note in notes[:2]:
            lines.append(f"- {note}")
        lines.append("")
    if answer.strip():
        lines.extend(["## Takeaway", answer.strip(), ""])
    return payload, "\n".join(lines).strip()


def _format_comparison_table_artifact(query: str, profiles: list[dict]) -> tuple[dict, str]:
    rows = []
    for profile in profiles:
        rows.append(
            {
                "paper_id": profile.get("paper_id", ""),
                "title": profile.get("title", ""),
                "problem": profile.get("problem", ""),
                "core_method": profile.get("core_method", ""),
                "assumptions": ", ".join(profile.get("assumptions", [])[:3]),
            }
        )
    lines = [
        f"Question: {query}",
        "",
        "| Paper | Problem | Core Method | Assumptions |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['paper_id']} | {row['problem'] or row['title']} | {row['core_method'] or '-'} | {row['assumptions'] or '-'} |"
        )
    return {"kind": "build_comparison_table", "query": query, "rows": rows}, "\n".join(lines)


def _format_research_summary_artifact(query: str, answer: str, profiles: list[dict]) -> tuple[dict, str]:
    payload = {
        "kind": "save_research_summary",
        "query": query,
        "papers": [profile.get("paper_id", "") for profile in profiles],
        "summary": answer.strip(),
    }
    lines = [
        "# Research Summary Card",
        "",
        f"Topic: {query}",
        f"Papers: {', '.join(payload['papers'])}",
        "",
        "## Summary",
        answer.strip() or "No summary generated.",
    ]
    return payload, "\n".join(lines).strip()


def _apply_artifact_kind(state: State, answer: str, profiles: list[dict] | None = None, evidence: list[dict] | None = None) -> tuple[dict, str]:
    artifact_kind = state.get("artifact_kind", "") or ""
    profiles = profiles or []
    evidence = evidence or []
    if artifact_kind == "export_notes":
        return _format_notes_artifact(state.get("originalQuery", ""), answer, evidence)
    if artifact_kind == "build_comparison_table":
        return _format_comparison_table_artifact(state.get("originalQuery", ""), profiles)
    if artifact_kind == "save_research_summary":
        return _format_research_summary_artifact(state.get("originalQuery", ""), answer, profiles)
    return {}, answer


def _get_or_build_paper_profile(workspace_id, paper_id, llm, parent_store, workspace_memory):
    parent_chunks = parent_store.load_paper(workspace_id, paper_id)
    profile_chunks = sorted(parent_chunks, key=_profile_parent_chunk_priority)[:8]
    source_parent_ids = [item.get("parent_id") for item in profile_chunks if item.get("parent_id")]
    source_sections = [
        item.get("metadata", {}).get("section", "")
        for item in profile_chunks
        if item.get("metadata", {}).get("section", "")
    ]

    existing = workspace_memory.load_paper_profile(workspace_id, paper_id)
    if existing:
        if source_parent_ids and not existing.get("source_parent_ids"):
            existing = {
                **existing,
                "source_parent_ids": source_parent_ids,
                "source_sections": source_sections,
            }
            workspace_memory.save_paper_profile(workspace_id, paper_id, existing)
        return existing

    if not parent_chunks:
        profile = PaperProfile(paper_id=paper_id).model_dump()
        workspace_memory.save_paper_profile(workspace_id, paper_id, profile)
        return profile

    content = "\n\n".join(
        f"Section: {item['metadata'].get('section', 'Unknown')}\n{item['content']}" for item in profile_chunks
    )
    profile_prompt = (
        f"Paper ID: {paper_id}\n\n"
        f"Paper content excerpts:\n{content}"
    )
    llm_with_structure = llm.with_structured_output(PaperProfile, method="function_calling")
    profile = llm_with_structure.invoke(
        [SystemMessage(content=get_paper_profile_prompt()), HumanMessage(content=profile_prompt)]
    )
    payload = profile.model_dump()
    if not payload.get("title"):
        payload["title"] = paper_id
    payload["source_parent_ids"] = source_parent_ids
    payload["source_sections"] = source_sections
    workspace_memory.save_paper_profile(workspace_id, paper_id, payload)
    workspace_memory.save_semantic_fact(
        workspace_id,
        f"paper_summary::{paper_id}",
        {
            "kind": "paper_summary",
            "paper_id": paper_id,
            "title": payload.get("title", ""),
            "core_method": payload.get("core_method", ""),
        },
    )
    return payload


def compare_papers(state: State, llm, parent_store, evidence_retriever, workspace_memory, default_workspace_id: str):
    workspace_id, paper_ids = _target_papers(state, workspace_memory, default_workspace_id)
    resolved_query = state.get("rewrittenQuestions", [state.get("originalQuery", "")])[0]
    support_query = _support_selection_query(state)
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    task_intent = state.get("task_intent") or retrieval_plan.get("task_intent", "multi_paper_comparison")
    answer_format = state.get("answer_format") or retrieval_plan.get("answer_format", "short_answer")
    profiles = [
        _get_or_build_paper_profile(workspace_id, paper_id, llm, parent_store, workspace_memory)
        for paper_id in paper_ids
    ]
    _, _, support_parent_chunks, merged_records, rerank_backend = _retrieve_support_groups(
        state,
        evidence_retriever,
        workspace_memory,
        default_workspace_id,
        support_query or resolved_query,
    )
    support_blocks = []
    for profile in profiles:
        paper_id = profile.get("paper_id", "")
        selected_chunks = [item for item in support_parent_chunks if item.get("metadata", {}).get("paper_id") == paper_id]
        if not selected_chunks:
            selected_chunks = _select_supporting_parent_chunks(parent_store.load_paper(workspace_id, paper_id), support_query or resolved_query, limit=3)
        excerpts = "\n\n".join(
            f"Section: {item.get('metadata', {}).get('section', 'Unknown')}\n{item.get('content', '').strip()}"
            for item in selected_chunks[:3]
        )
        support_blocks.append(f"Paper: {paper_id}\nRelevant excerpts:\n{excerpts}")
    prompt_content = (
        f"Original user question:\n{state.get('originalQuery')}\n\n"
        f"Task intent:\n{task_intent}\n\n"
        f"Preferred answer format:\n{answer_format}\n\n"
        f"Response language:\n{_response_language_instruction(state)}\n\n"
        f"Resolved query:\n{resolved_query}\n\n"
        f"Paper profiles:\n{profiles}\n\n"
        "Generation guidance:\n"
        "- Build per-paper evidence cards before drawing cross-paper conclusions.\n"
        "- Keep paper attribution explicit for each claim.\n"
        + ("- Because this is citation finding, output evidence items with paper, claim, section, snippet, and relevance.\n" if task_intent == "citation_finding" or answer_format == "evidence_list" else "")
        + f"\nQuery-relevant supporting excerpts:\n\n" + "\n\n---\n\n".join(support_blocks)
    )
    response = llm.invoke([SystemMessage(content=get_comparison_prompt()), HumanMessage(content=prompt_content)])
    artifact_payload, final_answer = _apply_artifact_kind(state, response.content, profiles=profiles, evidence=merged_records)
    return {
        "messages": [AIMessage(content=final_answer)],
        "paper_profiles": profiles,
        "final_answer": final_answer,
        "artifact_payload": artifact_payload,
        "retrieved_chunks": [],
        "retrieved_parent_chunks": [_record_from_parent_chunk(item) for item in support_parent_chunks],
        "retrieved_paper_ids": paper_ids,
        "retrieved_sections": _unique_non_empty(
            [item.get("metadata", {}).get("section", "") for item in support_parent_chunks]
        ),
        "rerank_backend": rerank_backend,
    }


def literature_review(state: State, llm, parent_store, evidence_retriever, workspace_memory, default_workspace_id: str):
    workspace_id, paper_ids = _target_papers(state, workspace_memory, default_workspace_id)
    original_query = state.get("originalQuery", "")
    support_query = _support_selection_query(state) or original_query
    retrieval_plan = state.get("retrieval_plan", {}) or {}
    task_intent = state.get("task_intent") or retrieval_plan.get("task_intent", "literature_review")
    answer_format = state.get("answer_format") or retrieval_plan.get("answer_format", "related_work_paragraph")
    profiles = [
        _get_or_build_paper_profile(workspace_id, paper_id, llm, parent_store, workspace_memory)
        for paper_id in paper_ids
    ]
    _, _, support_parent_chunks, merged_records, rerank_backend = _retrieve_support_groups(
        state,
        evidence_retriever,
        workspace_memory,
        default_workspace_id,
        support_query,
    )
    support_blocks = []
    for profile in profiles:
        paper_id = profile.get("paper_id", "")
        selected_chunks = [item for item in support_parent_chunks if item.get("metadata", {}).get("paper_id") == paper_id]
        if not selected_chunks:
            selected_chunks = _select_supporting_parent_chunks(parent_store.load_paper(workspace_id, paper_id), support_query, limit=3)
        excerpts = "\n\n".join(
            f"Section: {item.get('metadata', {}).get('section', 'Unknown')}\n{item.get('content', '').strip()}"
            for item in selected_chunks[:3]
        )
        support_blocks.append(f"Paper: {paper_id}\nRelevant excerpts:\n{excerpts}")
    prompt_content = (
        f"Original user question:\n{original_query}\n\n"
        f"Task intent:\n{task_intent}\n\n"
        f"Preferred answer format:\n{answer_format}\n\n"
        f"Response language:\n{_response_language_instruction(state)}\n\n"
        f"Paper profiles:\n{profiles}\n\n"
        "Generation guidance:\n"
        "- Treat this as paper-level synthesis first, then use excerpts for claim grounding.\n"
        "- Organize by themes, methods, taxonomy, or timeline when appropriate.\n"
        + ("- Because this is citation finding, prioritize evidence items over long-form synthesis.\n" if task_intent == "citation_finding" or answer_format == "evidence_list" else "")
        + f"\nQuery-relevant supporting excerpts:\n\n" + "\n\n---\n\n".join(support_blocks)
    )
    response = llm.invoke(
        [SystemMessage(content=get_literature_review_prompt()), HumanMessage(content=prompt_content)]
    )
    artifact_payload, final_answer = _apply_artifact_kind(state, response.content, profiles=profiles, evidence=merged_records)
    return {
        "messages": [AIMessage(content=final_answer)],
        "paper_profiles": profiles,
        "final_answer": final_answer,
        "artifact_payload": artifact_payload,
        "retrieved_chunks": [],
        "retrieved_parent_chunks": [_record_from_parent_chunk(item) for item in support_parent_chunks],
        "retrieved_paper_ids": paper_ids,
        "retrieved_sections": _unique_non_empty(
            [item.get("metadata", {}).get("section", "") for item in support_parent_chunks]
        ),
        "rerank_backend": rerank_backend,
    }


def reflect_answer(state: State, llm, workspace_memory, default_workspace_id: str):
    if state.get("intent_type") not in {"cross_doc_comparison", "literature_review"}:
        return {}

    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    profiles = state.get("paper_profiles") or workspace_memory.list_paper_profiles(workspace_id)
    draft = state.get("final_answer") or ""
    if not draft:
        return {}

    supporting_excerpts = []
    for item in (state.get("retrieved_parent_chunks") or [])[:6]:
        supporting_excerpts.append(
            "\n".join(
                [
                    f"Paper: {item.get('paper_id', '')}",
                    f"Section: {item.get('section', 'Unknown')}",
                    f"Source: {item.get('source', 'Unknown')}",
                    f"Content: {item.get('content', '').strip()}",
                ]
            )
        )
    prompt_content = (
        f"Response language:\n{_response_language_instruction(state)}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Paper profiles:\n{profiles}\n\n"
        "Query-relevant supporting excerpts:\n\n"
        + ("\n\n---\n\n".join(supporting_excerpts) if supporting_excerpts else "None provided.")
    )
    response = llm.invoke([SystemMessage(content=get_reflection_prompt()), HumanMessage(content=prompt_content)])
    artifact_payload, final_answer = _apply_artifact_kind(
        state,
        response.content,
        profiles=profiles,
        evidence=state.get("retrieved_parent_chunks", []) or [],
    )
    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "artifact_payload": artifact_payload,
        "paper_profiles": profiles,
        "retrieved_parent_chunks": state.get("retrieved_parent_chunks", []) or [],
        "retrieved_chunks": state.get("retrieved_chunks", []) or [],
        "retrieved_paper_ids": state.get("retrieved_paper_ids", []) or [],
        "retrieved_sections": state.get("retrieved_sections", []) or [],
        "rerank_backend": state.get("rerank_backend", ""),
    }


def verify_answer(state: State, llm):
    draft = str(state.get("final_answer", "") or "").strip()
    if not draft:
        return {}

    evidence = state.get("retrieved_parent_chunks") or state.get("retrieved_chunks") or []
    source_names = _unique_non_empty([item.get("source", "") for item in evidence])
    support_summary = _evidence_support_summary(state.get("originalQuery", ""), evidence)
    fallback_answer = _standard_fallback_answer(state)
    is_multi_doc_synthesis = state.get("intent_type") in {"cross_doc_comparison", "literature_review"}

    if _is_fallback_answer(draft) or (
        not is_multi_doc_synthesis and support_summary["terms"] and not support_summary["matched_terms"]
    ):
        return {
            "messages": [AIMessage(content=fallback_answer)],
            "final_answer": fallback_answer,
            "verification_status": "downgrade",
        }

    if (
        state.get("intent_type") == "single_doc_close_reading"
        and support_summary["matched_terms"]
        and support_summary["support_ratio"] >= 0.5
    ):
        final_answer = draft
        if source_names and "Sources" not in final_answer:
            final_answer = f"{final_answer}\n\nSources\n" + "\n".join(source_names)
        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            "verification_status": "pass",
        }

    evidence_blocks = [
        "\n".join(
            [
                f"Paper: {item.get('paper_id', '')}",
                f"Section: {item.get('section', 'Unknown')}",
                f"Pages: {item.get('pages', '?')}",
                f"Source: {item.get('source', '')}",
                f"Content Type: {item.get('content_type', 'paragraph')}",
                f"Content: {item.get('content', '').strip()}",
            ]
        )
        for item in evidence[:4]
    ]

    if not evidence_blocks:
        return {
            "messages": [AIMessage(content=fallback_answer)],
            "final_answer": fallback_answer,
            "verification_status": "downgrade",
        }

    prompt_content = (
        f"User question:\n{state.get('originalQuery', '')}\n\n"
        f"Response language:\n{_response_language_instruction(state)}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Evidence blocks:\n\n" + "\n\n---\n\n".join(evidence_blocks)
    )
    try:
        verification = llm.with_structured_output(AnswerVerification, method="function_calling").invoke(
            [SystemMessage(content=get_verification_prompt()), HumanMessage(content=prompt_content)]
        )
        final_answer = verification.revised_answer.strip() or draft
        status = verification.verification_status
    except Exception:
        final_answer = draft
        status = "pass"

    if source_names and "Sources" not in final_answer:
        final_answer = f"{final_answer}\n\nSources\n" + "\n".join(source_names)

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "verification_status": status,
    }


def finalize_interaction(state: State, workspace_memory, default_workspace_id: str):
    workspace_context = state.get("workspace_context", {}) or {}
    workspace_id = workspace_context.get("workspace_id", default_workspace_id)
    status = state.get("verification_status") or "unknown"
    profile_paper_ids = [profile.get("paper_id", "") for profile in state.get("paper_profiles", []) or []]
    paper_ids = _unique_non_empty(
        (state.get("retrieved_paper_ids") or [])
        + (state.get("target_papers") or [])
        + (state.get("comparison_targets") or [])
        + (state.get("referenced_documents") or [])
        + profile_paper_ids
    )
    workspace_memory.record_interaction(
        workspace_id,
        {
            "intent_type": state.get("intent_type"),
            "task_intent": state.get("task_intent"),
            "query": state.get("originalQuery"),
            "paper_ids": paper_ids,
            "retrieved_sections": state.get("retrieved_sections", []) or [],
            "verification_status": status,
            "artifact_kind": state.get("artifact_kind", ""),
        },
    )
    return {"verification_status": status}


def aggregate_answers(state: State, llm, workspace_memory, default_workspace_id: str):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")], "final_answer": "No answers were generated."}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])
    rerank_backend = next(
        (item.get("rerank_backend", "") for item in sorted_answers if item.get("rerank_backend")),
        state.get("rerank_backend", ""),
    )
    if len(sorted_answers) == 1:
        answer = sorted_answers[0]["answer"]
        artifact_payload, answer = _apply_artifact_kind(
            state,
            answer,
            evidence=state.get("retrieved_parent_chunks", []) or state.get("retrieved_chunks", []) or [],
        )
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "artifact_payload": artifact_payload,
            "rerank_backend": rerank_backend,
        }

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"

    user_message = HumanMessage(
        content=(
            f"Response language: {_response_language_instruction(state)}\n"
            f"Original user question: {state['originalQuery']}\n"
            f"Retrieved answers:{formatted_answers}"
        )
    )
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    artifact_payload, final_answer = _apply_artifact_kind(
        state,
        synthesis_response.content,
        evidence=state.get("retrieved_parent_chunks", []) or state.get("retrieved_chunks", []) or [],
    )
    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "artifact_payload": artifact_payload,
        "rerank_backend": rerank_backend,
    }
