from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.settings import get_settings

from eval.judges import EvalJudge, calibration_to_dict

ROOT_DIR = Path(__file__).resolve().parents[1]

FALLBACK_MARKERS = [
    "insufficient",
    "missing from the current workspace",
    "current workspace",
    "not enough supporting passages",
    "could not find enough supporting passages",
    "no data was retrieved from the workspace",
    "not in the current workspace",
    "missing evidence",
    "evidence is insufficient",
    "\u672a\u68c0\u7d22\u5230",
    "\u8bc1\u636e\u4e0d\u8db3",
    "\u5f53\u524d\u5de5\u4f5c\u533a",
    "\u65e0\u6cd5",
    "\u627e\u4e0d\u5230",
]

DEFAULT_GATE_THRESHOLDS = {
    "overall_pass_rate": 0.70,
    "intent_routing": 0.85,
    "document_isolation": 0.95,
    "valid_fallback": 0.90,
    "evidence_hit_at_5": 0.50,
    "section_targeting": 0.70,
    "faithfulness": 0.60,
    "cross_doc_alignment": 0.60,
}

GATE_LABELS = {
    "overall_pass_rate": "Overall Pass Rate",
    "intent_routing": "Intent Routing Accuracy",
    "document_isolation": "Workspace / Document Isolation Rate",
    "valid_fallback": "Valid Fallback Rate",
    "evidence_hit_at_5": "Evidence Span Hit Rate@5",
    "section_targeting": "Section Targeting Accuracy",
    "faithfulness": "PaperProfile Faithfulness",
    "cross_doc_alignment": "Cross-Doc Alignment / Conflict Handling",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the literature-agent evaluation suite.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "eval" / "golden_dataset_v2.jsonl"))
    parser.add_argument("--output-json", default=str(ROOT_DIR / "eval" / "results" / "latest_results.json"))
    parser.add_argument("--output-report", default=str(ROOT_DIR / "eval" / "results" / "latest_report.md"))
    parser.add_argument("--template", default=str(ROOT_DIR / "eval" / "report_template.md.j2"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--baseline", choices=["full", "no_reflection"], default="full")
    parser.add_argument("--ablation-baseline", choices=["no_reflection"], default="")
    parser.add_argument("--min-pass-rate", type=float, default=DEFAULT_GATE_THRESHOLDS["overall_pass_rate"])
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Override an eval gate threshold, e.g. evidence_hit_at_5=0.60.",
    )
    parser.add_argument("--no-fail-on-threshold", action="store_true")
    return parser.parse_args()


HARD_METRIC_LABELS = {
    "intent_routing": "Intent Routing Accuracy",
    "paper_coverage": "Target Paper Coverage",
    "document_isolation": "Workspace / Document Isolation Rate",
    "valid_fallback": "Valid Fallback Rate",
    "evidence_hit_at_3": "Evidence Span Hit Rate@3",
    "evidence_hit_at_5": "Evidence Span Hit Rate@5",
    "section_targeting": "Section Targeting Accuracy",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict]:
    items = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            items.append(json.loads(line))
    return items


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").lower().split())


def normalize_section_name(text: str) -> str:
    cleaned = str(text or "").replace("_", " ").replace("*", " ").replace("`", " ")
    return " ".join(cleaned.lower().split())


def safe_excerpt(text: str, limit: int = 240) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in normalized)
    normalized = " ".join(normalized.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def parse_min_metric_overrides(items: list[str]) -> dict[str, float]:
    overrides = {}
    allowed = set(DEFAULT_GATE_THRESHOLDS)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --min-metric value {item!r}; expected metric=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(f"Unknown eval gate metric {key!r}. Allowed: {', '.join(sorted(allowed))}.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold for {key!r}: {raw_value!r}.") from exc
        if not 0 <= value <= 1:
            raise ValueError(f"Threshold for {key!r} must be between 0 and 1.")
        overrides[key] = value
    return overrides


def extract_token_usage(messages) -> dict[str, Any] | None:
    total_input = 0
    total_output = 0
    has_usage = False

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        usage = getattr(message, "usage_metadata", None) or {}
        if usage:
            total_input += int(usage.get("input_tokens", 0) or 0)
            total_output += int(usage.get("output_tokens", 0) or 0)
            has_usage = True
            continue

        response_metadata = getattr(message, "response_metadata", None) or {}
        if "token_usage" in response_metadata:
            token_usage = response_metadata.get("token_usage", {}) or {}
            total_input += int(token_usage.get("prompt_tokens", 0) or 0)
            total_output += int(token_usage.get("completion_tokens", 0) or 0)
            has_usage = True
            continue

        if "prompt_eval_count" in response_metadata or "eval_count" in response_metadata:
            total_input += int(response_metadata.get("prompt_eval_count", 0) or 0)
            total_output += int(response_metadata.get("eval_count", 0) or 0)
            has_usage = True

    if not has_usage:
        return None

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


def run_turn(rag_system: RAGSystem, workspace_id: str, focus_paper_id: str | None, query: str) -> dict[str, Any]:
    rag_system.set_workspace_context(workspace_id, focus_paper_id=focus_paper_id)
    config_obj = rag_system.get_config()
    current_state = rag_system.agent_graph.get_state(config_obj)
    working_memory = rag_system.workspace_memory.load_working_memory_snapshot(workspace_id)
    state_update = {
        "messages": [HumanMessage(content=query.strip())],
        "workspace_context": rag_system.get_workspace_context()["workspace_context"],
        "working_memory": working_memory,
    }

    invoke_input = state_update
    if current_state.next:
        rag_system.agent_graph.update_state(config_obj, state_update)
        invoke_input = None

    started_at = time.perf_counter()
    error = ""
    try:
        rag_system.agent_graph.invoke(invoke_input, config=config_obj)
    except Exception as exc:
        error = str(exc)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    final_state = rag_system.agent_graph.get_state(config_obj)
    values = final_state.values or {}
    workspace_context = values.get("workspace_context", {}) or {}
    messages = values.get("messages", []) or []
    final_answer = values.get("final_answer", "") or ""
    if final_state.next and not final_answer:
        final_answer = values.get("clarification_question", "") or "Interrupted before clarification."

    return {
        "status": "error" if error else ("interrupted" if final_state.next else "ok"),
        "error": error,
        "latency_ms": latency_ms,
        "predicted_intent": values.get("intent_type", ""),
        "final_answer": final_answer,
        "clarification_question": values.get("clarification_question", ""),
        "verification_status": values.get("verification_status", ""),
        "rerank_backend": values.get("rerank_backend", ""),
        "used_workspace_id": workspace_context.get("workspace_id", workspace_id),
        "used_focus_paper_id": workspace_context.get("focus_paper_id", focus_paper_id),
        "referenced_documents": values.get("referenced_documents", []) or [],
        "comparison_targets": values.get("comparison_targets", []) or [],
        "retrieved_chunks": values.get("retrieved_chunks", []) or [],
        "retrieved_parent_chunks": values.get("retrieved_parent_chunks", []) or [],
        "retrieved_paper_ids": values.get("retrieved_paper_ids", []) or [],
        "retrieved_sections": values.get("retrieved_sections", []) or [],
        "paper_profiles": values.get("paper_profiles", []) or [],
        "token_usage": extract_token_usage(messages),
    }


def answer_mentions_forbidden(answer: str, forbidden_paper_ids: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    return any(normalize_text(paper_id) in normalized_answer for paper_id in forbidden_paper_ids)


def _coerce_retrieved_records(turn_result: dict[str, Any]) -> list[dict[str, Any]]:
    records = turn_result.get("retrieved_parent_chunks") or turn_result.get("retrieved_chunks") or []
    normalized: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            normalized.append(record)
        elif isinstance(record, str):
            normalized.append({"content": record})
    return normalized


def _record_matches_gold(record: dict[str, Any], gold_entry: dict[str, Any]) -> bool:
    record_parent_id = str(record.get("parent_id", "") or "")
    gold_parent_id = str(gold_entry.get("parent_id", "") or "")
    if record_parent_id and gold_parent_id and record_parent_id == gold_parent_id:
        return True

    record_paper_id = normalize_text(record.get("paper_id", ""))
    gold_paper_id = normalize_text(gold_entry.get("paper_id", ""))
    if gold_paper_id and record_paper_id and record_paper_id != gold_paper_id:
        return False

    gold_section = normalize_section_name(gold_entry.get("section", ""))
    if gold_section:
        record_section = normalize_section_name(record.get("section", ""))
        if record_section and record_section != gold_section:
            if not record_parent_id:
                return False

    content = normalize_text(record.get("content", ""))
    if not content:
        return False

    match_any = gold_entry.get("match_any", []) or []
    if not match_any:
        return False
    return any(normalize_text(snippet) in content for snippet in match_any)


def _evidence_hit(records: list[dict[str, Any]], gold_evidence: list[dict[str, Any]], top_k: int) -> bool | None:
    if not gold_evidence:
        return None

    window = records[:top_k]
    if not window:
        return False

    return all(any(_record_matches_gold(record, gold_entry) for record in window) for gold_entry in gold_evidence)


def _failure_reason(hard_metrics: dict[str, bool | None]) -> str:
    labels = []
    for key, value in hard_metrics.items():
        if value is False:
            labels.append(f"{key}_failed")
    return "; ".join(labels)


def _requested_judge_rubrics(case: dict) -> set[str]:
    aliases = {
        "faithfulness": "faithfulness",
        "alignment": "cross_doc_alignment",
        "cross_doc_alignment": "cross_doc_alignment",
        "valid_fallback": "valid_fallback",
    }
    requested = case.get("judge_rubrics")
    if requested is None:
        if case.get("fallback_expected"):
            return {"valid_fallback"}
        if case.get("intent_label") == "cross_doc_comparison":
            return {"cross_doc_alignment"}
        return {"faithfulness"}
    normalized = set()
    for item in requested or []:
        key = aliases.get(str(item or "").strip().lower())
        if key:
            normalized.add(key)
    return normalized


def evaluate_case(case: dict, turn_result: dict, judge: EvalJudge | None) -> dict[str, Any]:
    answer = turn_result.get("final_answer", "")
    retrieved_papers = turn_result.get("retrieved_paper_ids", []) or []
    retrieved_sections = turn_result.get("retrieved_sections", []) or []
    expected_paper_ids = case.get("expected_paper_ids", []) or []
    expected_sections = case.get("expected_sections", []) or []
    forbidden_paper_ids = case.get("forbidden_paper_ids", []) or []
    gold_evidence = case.get("gold_evidence", []) or []
    retrieved_records = _coerce_retrieved_records(turn_result)

    intent_match = turn_result.get("predicted_intent") == case.get("intent_label")
    paper_coverage = all(paper_id in retrieved_papers or paper_id in turn_result.get("referenced_documents", []) for paper_id in expected_paper_ids)
    normalized_retrieved_sections = [normalize_section_name(item) for item in retrieved_sections]
    section_hit = any(normalize_section_name(section) in normalized_retrieved_sections for section in expected_sections) if expected_sections else None
    fallback_detected = any(marker in normalize_text(answer) for marker in [normalize_text(item) for item in FALLBACK_MARKERS])
    forbidden_hit = answer_mentions_forbidden(answer, forbidden_paper_ids)
    fallback_expected = bool(case.get("fallback_expected"))
    retrieval_isolation = None
    answer_isolation = None
    document_isolation = None
    if case.get("intent_label") == "single_doc_close_reading":
        retrieval_isolation = not any(paper_id in retrieved_papers for paper_id in forbidden_paper_ids)
        answer_isolation = not forbidden_hit
        document_isolation = retrieval_isolation and answer_isolation

    evidence_hit_at_3 = _evidence_hit(retrieved_records, gold_evidence, 3)
    evidence_hit_at_5 = _evidence_hit(retrieved_records, gold_evidence, 5)

    requested_judges = _requested_judge_rubrics(case)
    judge_scores = {}
    if judge and judge.available:
        if "valid_fallback" in requested_judges:
            score = judge.judge_valid_fallback(
                case.get("user_query", ""),
                answer,
                turn_result.get("used_workspace_id", case.get("workspace_id", "")),
                retrieved_papers,
            )
            if score:
                judge_scores["valid_fallback"] = score.as_dict()
        if "cross_doc_alignment" in requested_judges and turn_result.get("paper_profiles"):
            score = judge.judge_alignment(case.get("user_query", ""), turn_result.get("paper_profiles", []), answer)
            if score:
                judge_scores["cross_doc_alignment"] = score.as_dict()
        if "faithfulness" in requested_judges and turn_result.get("paper_profiles"):
            score = judge.judge_faithfulness(case.get("user_query", ""), turn_result.get("paper_profiles", []), answer)
            if score:
                judge_scores["faithfulness"] = score.as_dict()

    hard_metrics = {
        "intent_routing": intent_match,
        "paper_coverage": paper_coverage if expected_paper_ids else None,
        "retrieval_isolation": retrieval_isolation,
        "answer_isolation": answer_isolation,
        "document_isolation": document_isolation,
        "valid_fallback": fallback_detected if fallback_expected else None,
        "evidence_hit_at_3": evidence_hit_at_3,
        "evidence_hit_at_5": evidence_hit_at_5,
        "section_targeting": section_hit,
    }
    judge_metrics = {
        "faithfulness_judge": judge_scores.get("faithfulness", {}).get("pass")
        if "faithfulness" in requested_judges and judge_scores.get("faithfulness")
        else None,
        "cross_doc_alignment_judge": judge_scores.get("cross_doc_alignment", {}).get("pass")
        if "cross_doc_alignment" in requested_judges and judge_scores.get("cross_doc_alignment")
        else None,
        "valid_fallback_judge": judge_scores.get("valid_fallback", {}).get("pass")
        if "valid_fallback" in requested_judges and judge_scores.get("valid_fallback")
        else None,
    }
    failure_reason = _failure_reason({**hard_metrics, **judge_metrics})
    passed = turn_result.get("status") == "ok" and not failure_reason

    return {
        "case_id": case.get("case_id"),
        "suite": case.get("suite"),
        "notes": case.get("notes", ""),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "failure_reason": failure_reason,
        "query": case.get("user_query", ""),
        "intent_match": intent_match,
        "paper_coverage": paper_coverage,
        "section_hit": section_hit,
        "fallback_expected": fallback_expected,
        "fallback_detected": fallback_detected,
        "forbidden_hit": forbidden_hit,
        "latency_ms": turn_result.get("latency_ms"),
        "predicted_intent": turn_result.get("predicted_intent"),
        "expected_intent": case.get("intent_label", ""),
        "final_answer": answer,
        "answer": answer,
        "used_workspace_id": turn_result.get("used_workspace_id", case.get("workspace_id", "")),
        "used_focus_paper_id": turn_result.get("used_focus_paper_id"),
        "retrieved_chunks": turn_result.get("retrieved_chunks", []) or [],
        "retrieved_parent_chunks": turn_result.get("retrieved_parent_chunks", []) or [],
        "retrieved_paper_ids": retrieved_papers,
        "retrieved_sections": retrieved_sections,
        "paper_profiles": turn_result.get("paper_profiles", []) or [],
        "token_usage": turn_result.get("token_usage"),
        "history_turn_count": len(case.get("history", []) or []),
        "hard_metrics": hard_metrics,
        "judge_rubrics": sorted(requested_judges),
        "judge_metrics": judge_metrics,
        "judge_results": {
            "faithfulness": judge_scores.get("faithfulness"),
            "cross_doc_alignment": judge_scores.get("cross_doc_alignment"),
            "valid_fallback": judge_scores.get("valid_fallback"),
        },
        "judge_scores": judge_scores,
        "turn_result": turn_result,
    }


def run_dataset_cases(dataset: list[dict], settings, baseline: str, judge: EvalJudge | None) -> list[dict[str, Any]]:
    from agentic_rag.bootstrap import build_runtime

    runtime = build_runtime(settings, enable_reflection=(baseline != "no_reflection"))
    rag_system = runtime.rag_system

    results = []
    for case in dataset:
        rag_system.reset_thread()
        workspace_id = case.get("workspace_id", settings.default_workspace_id)
        focus_paper_id = case.get("focus_paper_id")

        for history_item in case.get("history", []) or []:
            if isinstance(history_item, dict):
                query = history_item.get("user", "")
            else:
                query = str(history_item or "")
            if query.strip():
                run_turn(rag_system, workspace_id, focus_paper_id, query)

        turn_result = run_turn(rag_system, workspace_id, focus_paper_id, case.get("user_query", ""))
        results.append(evaluate_case(case, turn_result, judge))

    return results


def _bool_rate(values: list[bool | None]) -> tuple[float | None, int, int]:
    eligible = [item for item in values if item is not None]
    if not eligible:
        return None, 0, 0
    passed = sum(1 for item in eligible if item is True)
    return passed / len(eligible), passed, len(eligible)


def build_thresholds(min_pass_rate: float, min_metric_items: list[str]) -> dict[str, float]:
    if not 0 <= min_pass_rate <= 1:
        raise ValueError("--min-pass-rate must be between 0 and 1.")
    thresholds = dict(DEFAULT_GATE_THRESHOLDS)
    thresholds["overall_pass_rate"] = min_pass_rate
    thresholds.update(parse_min_metric_overrides(min_metric_items))
    return thresholds


def build_eval_gate(results: list[dict], thresholds: dict[str, float]) -> dict[str, Any]:
    rows = []
    total = len(results)
    overall_rate = (sum(1 for item in results if item.get("passed")) / total) if total else None

    def add_row(key: str, actual: float | None, passed_count: int, eligible: int):
        threshold = thresholds[key]
        skipped = actual is None
        passed = True if skipped else actual >= threshold
        rows.append(
            {
                "key": key,
                "label": GATE_LABELS[key],
                "threshold": threshold,
                "actual": actual,
                "passed_count": passed_count,
                "eligible": eligible,
                "passed": passed,
                "reason": "No eligible cases." if skipped else ("" if passed else "Below threshold."),
            }
        )

    add_row("overall_pass_rate", overall_rate, sum(1 for item in results if item.get("passed")), total)
    for key in ["intent_routing", "document_isolation", "valid_fallback", "evidence_hit_at_5", "section_targeting"]:
        actual, passed_count, eligible = _bool_rate([item.get("hard_metrics", {}).get(key) for item in results])
        add_row(key, actual, passed_count, eligible)

    for key in ["faithfulness", "cross_doc_alignment"]:
        actual, passed_count, eligible = _bool_rate(
            [
                (item.get("judge_results", {}).get(key) or {}).get("pass")
                for item in results
                if item.get("judge_results", {}).get(key)
            ]
        )
        add_row(key, actual, passed_count, eligible)

    failed_rows = [row for row in rows if row["passed"] is False]
    return {
        "passed": not failed_rows,
        "rows": rows,
        "failure_reasons": [f"{row['key']}: {row['reason']}" for row in failed_rows],
        "thresholds": thresholds,
    }


def build_report_payload(results: list[dict], metadata: dict[str, Any], dataset: list[dict]) -> dict[str, Any]:
    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for item in results if item["passed"]),
        "failed_cases": sum(1 for item in results if not item["passed"]),
        "avg_latency_ms": round(sum(item.get("latency_ms", 0) or 0 for item in results) / max(len(results), 1), 2),
    }

    def metric_summary(key: str, label: str) -> dict[str, Any]:
        eligible_items = [item for item in results if item.get("hard_metrics", {}).get(key) is not None]
        passed_items = [item for item in eligible_items if item["hard_metrics"].get(key) is True]
        eligible = len(eligible_items)
        passed = len(passed_items)
        rate = (passed / eligible) if eligible else None
        return {"label": label, "eligible": eligible, "passed": passed, "rate": rate}

    hard_metrics = {key: metric_summary(key, label) for key, label in HARD_METRIC_LABELS.items()}

    per_intent = {}
    for intent in sorted({item.get("expected_intent", "") for item in results}):
        scoped = [item for item in results if item.get("expected_intent", "") == intent]
        if not scoped:
            continue
        passed = sum(1 for item in scoped if item.get("intent_match"))
        per_intent[intent] = {
            "eligible": len(scoped),
            "passed": passed,
            "rate": passed / len(scoped) if scoped else None,
        }

    failure_distribution = Counter()
    for item in results:
        failure_reason = item.get("failure_reason", "")
        if not failure_reason:
            continue
        for label in [part.strip() for part in failure_reason.split(";") if part.strip()]:
            failure_distribution[label] += 1

    def judge_metric_summary(key: str, label: str) -> dict[str, Any]:
        eligible_items = [item for item in results if item.get("judge_results", {}).get(key)]
        scores = [item["judge_results"][key]["score"] for item in eligible_items]
        passed_items = [item for item in eligible_items if item["judge_results"][key].get("pass") is True]
        eligible = len(eligible_items)
        passed = len(passed_items)
        return {
            "label": label,
            "eligible": eligible,
            "passed": passed,
            "rate": (passed / eligible) if eligible else None,
            "average_score": round(sum(scores) / len(scores), 2) if scores else None,
        }

    soft_metrics = {
        "faithfulness": judge_metric_summary("faithfulness", "PaperProfile Faithfulness"),
        "cross_doc_alignment": judge_metric_summary("cross_doc_alignment", "Cross-Doc Alignment / Conflict Handling"),
        "valid_fallback": judge_metric_summary("valid_fallback", "Judge-Validated Fallback"),
    }

    def report_example(item: dict) -> dict:
        return {**item, "final_answer_excerpt": safe_excerpt(item.get("final_answer", ""))}

    success_examples = [report_example(item) for item in results if item["passed"]][:2]
    failure_examples = [report_example(item) for item in results if not item["passed"]][:2]
    calibration = metadata.get("calibration") or {
        "available": False,
        "calibration_passed": False,
        "reason": "Skipped.",
        "details": {},
    }

    ablation_results = metadata.get("ablation_results") or []

    def ablation_rate(key: str) -> float | None:
        eligible_items = [item for item in ablation_results if item.get("hard_metrics", {}).get(key) is not None]
        if not eligible_items:
            return None
        passed_items = [item for item in eligible_items if item["hard_metrics"].get(key) is True]
        return len(passed_items) / len(eligible_items)

    ablation_comparison = None
    if ablation_results:
        ablation_comparison = {
            "primary_baseline": metadata["baseline"],
            "secondary_baseline": metadata.get("ablation_baseline", ""),
            "rows": [
                {
                    "metric": label,
                    "primary_rate": hard_metrics[key]["rate"],
                    "secondary_rate": ablation_rate(key),
                }
                for key, label in HARD_METRIC_LABELS.items()
            ],
            "primary_avg_latency_ms": summary["avg_latency_ms"],
            "secondary_avg_latency_ms": round(
                sum(item.get("latency_ms", 0) or 0 for item in ablation_results) / max(len(ablation_results), 1),
                2,
            ),
        }

    return {
        "generated_at": metadata["generated_at"],
        "dataset": {
            "path": metadata["dataset_path"],
            "case_count": len(dataset),
            "suite_counts": dict(Counter(case.get("suite", "unknown") for case in dataset)),
            "intent_counts": dict(Counter(case.get("intent_label", "unknown") for case in dataset)),
            "workspace_counts": metadata.get("workspace_counts", {}),
        },
        "primary": {
            "generated_at": metadata["generated_at"],
            "baseline": metadata["baseline"],
            "case_count": len(results),
            "metrics": {
                "hard_metrics": hard_metrics,
                "soft_metrics": soft_metrics,
                "per_intent": per_intent,
                "failure_distribution": dict(failure_distribution),
                "success_examples": success_examples,
                "failure_examples": failure_examples,
                "average_latency_ms": summary["avg_latency_ms"],
            },
            "calibration": calibration,
            "gate": metadata.get("gate") or {"passed": True, "rows": [], "failure_reasons": [], "thresholds": {}},
        },
        "environment": {
            "llm_provider": metadata["llm_provider"],
            "llm_model": metadata["llm_model"],
            "embedding_provider": metadata["embedding_provider"],
            "embedding_model": metadata["embedding_model"],
            "judge_enabled": metadata.get("judge_enabled", False),
            "git_commit": metadata.get("git_commit", ""),
            "rerank_backend_counts": metadata.get("rerank_backend_counts", {}),
            "verification_status_counts": metadata.get("verification_status_counts", {}),
        },
        "ablation": bool(ablation_comparison),
        "ablation_comparison": ablation_comparison,
    }


def render_report(results: list[dict], metadata: dict[str, Any], dataset: list[dict], template_path: Path) -> str:
    payload = build_report_payload(results, metadata, dataset)
    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for item in results if item["passed"]),
        "failed_cases": sum(1 for item in results if not item["passed"]),
        "avg_latency_ms": round(sum(item.get("latency_ms", 0) or 0 for item in results) / max(len(results), 1), 2),
    }

    def percentage(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.1f}%"

    if template_path.exists():
        template = Template(template_path.read_text(encoding="utf-8"))
        return template.render(payload=payload, metadata=metadata, summary=summary, results=results, percentage=percentage)

    lines = [
        "# Evaluation Report",
        "",
        f"- Generated at: {metadata['generated_at']}",
        f"- Baseline: {metadata['baseline']}",
        f"- Cases: {summary['total_cases']}",
        f"- Passed: {summary['passed_cases']}",
        f"- Failed: {summary['failed_cases']}",
        f"- Avg latency (ms): {summary['avg_latency_ms']}",
        f"- Gate passed: {metadata.get('gate', {}).get('passed', True)}",
        "",
        "## Rerank Backends",
        "",
    ]
    for backend, count in (metadata.get("rerank_backend_counts") or {}).items():
        lines.append(f"- {backend}: {count}")
    lines.extend(
        [
            "",
        "## Cases",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"### {result['case_id']}",
                f"- Passed: {result['passed']}",
                f"- Intent match: {result.get('intent_match')}",
                f"- Paper coverage: {result.get('paper_coverage')}",
                f"- Section hit: {result.get('section_hit')}",
                f"- Fallback detected: {result.get('fallback_detected')}",
                f"- Forbidden hit: {result.get('forbidden_hit')}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        thresholds = build_thresholds(args.min_pass_rate, args.min_metric)
    except ValueError as exc:
        print(f"Evaluation argument error: {exc}", file=sys.stderr)
        return 2

    settings = get_settings()
    dataset = load_jsonl(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]

    judge = None if args.skip_judge else EvalJudge()
    calibration = None
    if judge and judge.available and not args.skip_calibration:
        calibration = calibration_to_dict(judge.run_calibration())

    results = run_dataset_cases(dataset, settings, args.baseline, judge)
    ablation_results = []
    if args.ablation_baseline:
        ablation_results = run_dataset_cases(dataset, settings, args.ablation_baseline, judge)

    gate = build_eval_gate(results, thresholds)
    metadata = {
        "generated_at": utc_now(),
        "git_commit": get_git_commit(),
        "baseline": args.baseline,
        "ablation_baseline": args.ablation_baseline,
        "dataset_path": str(Path(args.dataset)),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "judge_enabled": bool(judge and judge.available),
        "workspace_counts": dict(Counter(case.get("workspace_id", settings.default_workspace_id) for case in dataset)),
        "rerank_backend_counts": dict(Counter(result["turn_result"].get("rerank_backend", "") or "unknown" for result in results)),
        "verification_status_counts": dict(Counter(result["turn_result"].get("verification_status", "") or "unknown" for result in results)),
        "calibration": calibration,
        "gate": gate,
    }
    report_metadata = {**metadata, "ablation_results": ablation_results}

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {"metadata": metadata, "results": results, "ablation_results": ablation_results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    report_text = render_report(results, report_metadata, dataset, Path(args.template))
    output_report.write_text(report_text, encoding="utf-8")

    passed = sum(1 for item in results if item["passed"])
    print(f"Evaluation completed: {passed}/{len(results)} cases passed.")
    print(f"Gate: {'passed' if gate['passed'] else 'failed'}")
    for reason in gate.get("failure_reasons", []):
        print(f"- {reason}")
    print(f"JSON results: {output_json}")
    print(f"Report: {output_report}")
    return 0 if gate["passed"] or args.no_fail_on_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
