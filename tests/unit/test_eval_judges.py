import unittest

from eval import evaluate as eval_module


class DummyScore:
    def __init__(self, score, passed, reason=""):
        self.score = score
        self.passed = passed
        self.reason = reason

    def as_dict(self):
        return {"score": self.score, "pass": self.passed, "reason": self.reason}


class FakeJudge:
    available = True

    def __init__(self, faithfulness=None, alignment=None, valid_fallback=None):
        self._faithfulness = faithfulness
        self._alignment = alignment
        self._valid_fallback = valid_fallback
        self.calls = {"faithfulness": 0, "alignment": 0, "valid_fallback": 0}

    def judge_faithfulness(self, user_query, paper_profiles, final_answer):
        self.calls["faithfulness"] += 1
        return self._faithfulness

    def judge_alignment(self, user_query, paper_profiles, final_answer):
        self.calls["alignment"] += 1
        return self._alignment

    def judge_valid_fallback(self, user_query, final_answer, workspace_id, retrieved_paper_ids):
        self.calls["valid_fallback"] += 1
        return self._valid_fallback


def _base_turn_result():
    return {
        "status": "ok",
        "predicted_intent": "single_doc_close_reading",
        "final_answer": "Supported answer.",
        "latency_ms": 10,
        "retrieved_paper_ids": ["paper-1"],
        "retrieved_sections": ["Intro"],
        "referenced_documents": ["paper-1"],
        "paper_profiles": [{"paper_id": "paper-1", "title": "Paper One"}],
        "retrieved_parent_chunks": [
            {
                "paper_id": "paper-1",
                "parent_id": "paper-1_parent_0",
                "section": "Intro",
                "content": "alpha beta gamma",
            }
        ],
    }


class EvalJudgeBehaviorTests(unittest.TestCase):
    def test_empty_judge_rubrics_disables_judge_calls(self):
        case = {
            "case_id": "case-1",
            "suite": "smoke",
            "user_query": "What happened?",
            "intent_label": "single_doc_close_reading",
            "expected_paper_ids": ["paper-1"],
            "expected_sections": ["Intro"],
            "forbidden_paper_ids": [],
            "gold_evidence": [
                {"paper_id": "paper-1", "parent_id": "paper-1_parent_0", "section": "Intro", "match_any": ["alpha"]}
            ],
            "judge_rubrics": [],
        }
        judge = FakeJudge(faithfulness=DummyScore(0, False, "should not run"))

        result = eval_module.evaluate_case(case, _base_turn_result(), judge)

        self.assertTrue(result["passed"])
        self.assertEqual(judge.calls, {"faithfulness": 0, "alignment": 0, "valid_fallback": 0})
        self.assertEqual(result["judge_rubrics"], [])
        self.assertEqual(result["judge_results"]["faithfulness"], None)

    def test_failed_requested_judge_fails_case(self):
        case = {
            "case_id": "case-2",
            "suite": "smoke",
            "user_query": "Summarize the paper.",
            "intent_label": "single_doc_close_reading",
            "expected_paper_ids": ["paper-1"],
            "expected_sections": ["Intro"],
            "forbidden_paper_ids": [],
            "gold_evidence": [
                {"paper_id": "paper-1", "parent_id": "paper-1_parent_0", "section": "Intro", "match_any": ["alpha"]}
            ],
            "judge_rubrics": ["faithfulness"],
        }
        judge = FakeJudge(faithfulness=DummyScore(3, False, "not faithful enough"))

        result = eval_module.evaluate_case(case, _base_turn_result(), judge)

        self.assertFalse(result["passed"])
        self.assertIn("faithfulness_judge_failed", result["failure_reason"])
        self.assertEqual(judge.calls["faithfulness"], 1)
        self.assertEqual(result["judge_results"]["faithfulness"]["pass"], False)

    def test_alignment_alias_is_supported(self):
        case = {
            "case_id": "case-3",
            "suite": "smoke",
            "user_query": "Compare the papers.",
            "intent_label": "cross_doc_comparison",
            "expected_paper_ids": ["paper-1"],
            "expected_sections": ["Intro"],
            "forbidden_paper_ids": [],
            "gold_evidence": [
                {"paper_id": "paper-1", "parent_id": "paper-1_parent_0", "section": "Intro", "match_any": ["alpha"]}
            ],
            "judge_rubrics": ["alignment"],
        }
        turn_result = _base_turn_result()
        turn_result["predicted_intent"] = "cross_doc_comparison"
        judge = FakeJudge(alignment=DummyScore(4, True, "aligned"))

        result = eval_module.evaluate_case(case, turn_result, judge)

        self.assertTrue(result["passed"])
        self.assertEqual(result["judge_rubrics"], ["cross_doc_alignment"])
        self.assertEqual(judge.calls["alignment"], 1)
        self.assertEqual(result["judge_results"]["cross_doc_alignment"]["pass"], True)

    def test_soft_metrics_use_judge_pass_flag(self):
        results = [
            {
                "passed": True,
                "latency_ms": 5,
                "expected_intent": "single_doc_close_reading",
                "intent_match": True,
                "failure_reason": "",
                "judge_results": {
                    "faithfulness": {"score": 3, "pass": False, "reason": "borderline"},
                    "cross_doc_alignment": None,
                    "valid_fallback": None,
                },
            }
        ]
        payload = eval_module.build_report_payload(
            results,
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "dataset_path": "eval/golden_dataset_v2.jsonl",
                "baseline": "full",
                "llm_provider": "test",
                "llm_model": "test-model",
                "embedding_provider": "test",
                "embedding_model": "embed-model",
                "judge_enabled": True,
                "workspace_counts": {"demo": 1},
                "rerank_backend_counts": {},
                "verification_status_counts": {},
                "calibration": None,
            },
            [{"suite": "smoke", "intent_label": "single_doc_close_reading"}],
        )

        faithfulness = payload["primary"]["metrics"]["soft_metrics"]["faithfulness"]
        self.assertEqual(faithfulness["eligible"], 1)
        self.assertEqual(faithfulness["passed"], 0)
        self.assertEqual(faithfulness["average_score"], 3.0)
        self.assertTrue(payload["environment"]["judge_enabled"])

    def test_ablation_payload_compares_secondary_baseline(self):
        primary_result = {
            "passed": True,
            "latency_ms": 10,
            "expected_intent": "cross_doc_comparison",
            "intent_match": True,
            "failure_reason": "",
            "hard_metrics": {
                "intent_routing": True,
                "paper_coverage": True,
                "document_isolation": None,
                "valid_fallback": None,
                "evidence_hit_at_3": True,
                "evidence_hit_at_5": True,
                "section_targeting": True,
            },
            "judge_results": {"faithfulness": None, "cross_doc_alignment": None, "valid_fallback": None},
        }
        secondary_result = {
            **primary_result,
            "passed": False,
            "latency_ms": 20,
            "hard_metrics": {
                **primary_result["hard_metrics"],
                "evidence_hit_at_3": False,
            },
        }

        payload = eval_module.build_report_payload(
            [primary_result],
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "dataset_path": "eval/golden_dataset_v2.jsonl",
                "baseline": "full",
                "ablation_baseline": "no_reflection",
                "llm_provider": "test",
                "llm_model": "test-model",
                "embedding_provider": "test",
                "embedding_model": "embed-model",
                "judge_enabled": False,
                "workspace_counts": {"demo": 1},
                "rerank_backend_counts": {},
                "verification_status_counts": {},
                "calibration": None,
                "ablation_results": [secondary_result],
            },
            [{"suite": "smoke", "intent_label": "cross_doc_comparison"}],
        )

        self.assertTrue(payload["ablation"])
        comparison = payload["ablation_comparison"]
        self.assertEqual(comparison["primary_baseline"], "full")
        self.assertEqual(comparison["secondary_baseline"], "no_reflection")
        evidence_row = next(row for row in comparison["rows"] if row["metric"] == "Evidence Span Hit Rate@3")
        self.assertEqual(evidence_row["primary_rate"], 1.0)
        self.assertEqual(evidence_row["secondary_rate"], 0.0)
        self.assertEqual(comparison["secondary_avg_latency_ms"], 20.0)

    def test_eval_gate_passes_when_rates_meet_thresholds(self):
        result = {
            "passed": True,
            "hard_metrics": {
                "intent_routing": True,
                "document_isolation": True,
                "valid_fallback": True,
                "evidence_hit_at_5": True,
                "section_targeting": True,
            },
            "judge_results": {
                "faithfulness": {"score": 4, "pass": True},
                "cross_doc_alignment": {"score": 4, "pass": True},
            },
        }

        gate = eval_module.build_eval_gate([result], eval_module.DEFAULT_GATE_THRESHOLDS)

        self.assertTrue(gate["passed"])
        self.assertFalse(gate["failure_reasons"])

    def test_eval_gate_fails_when_required_rate_is_below_threshold(self):
        result = {
            "passed": False,
            "hard_metrics": {
                "intent_routing": False,
                "document_isolation": True,
                "valid_fallback": None,
                "evidence_hit_at_5": False,
                "section_targeting": True,
            },
            "judge_results": {
                "faithfulness": {"score": 3, "pass": False},
                "cross_doc_alignment": None,
            },
        }

        gate = eval_module.build_eval_gate([result], eval_module.DEFAULT_GATE_THRESHOLDS)

        self.assertFalse(gate["passed"])
        self.assertIn("overall_pass_rate", gate["failure_reasons"][0])
        self.assertTrue(any("evidence_hit_at_5" in reason for reason in gate["failure_reasons"]))

    def test_threshold_overrides_and_no_fail_flag_inputs_are_supported(self):
        thresholds = eval_module.build_thresholds(
            0.5,
            ["intent_routing=0.75", "evidence_hit_at_5=0.25"],
        )

        self.assertEqual(thresholds["overall_pass_rate"], 0.5)
        self.assertEqual(thresholds["intent_routing"], 0.75)
        self.assertEqual(thresholds["evidence_hit_at_5"], 0.25)

    def test_safe_excerpt_collapses_control_characters_and_truncates(self):
        excerpt = eval_module.safe_excerpt("hello\n\nworld\x00" + "x" * 300, limit=20)

        self.assertEqual(excerpt, "hello world xxxxxxxx...")


if __name__ == "__main__":
    unittest.main()
