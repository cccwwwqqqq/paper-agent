import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agentic_rag.agents.schemas import IntentAnalysis
from agentic_rag.agents.edges import route_after_rewrite
from agentic_rag.agents.nodes import (
    _apply_artifact_kind,
    _build_retrieval_plan,
    close_reading,
    metadata_query_response,
    orchestrator,
    single_paper_summary,
    workspace_memory_qa,
    workspace_inventory_response,
)
from agentic_rag.settings import load_settings
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore


class P1NodeHelperTests(unittest.TestCase):
    def test_route_after_rewrite_sends_fine_grained_intents_to_closed_nodes(self):
        self.assertEqual(
            route_after_rewrite(
                {
                    "questionIsClear": True,
                    "intent_type": "general_retrieval",
                    "task_intent": "workspace_memory_qa",
                    "rewrittenQuestions": ["What did I read last time?"],
                }
            ),
            "workspace_memory_qa",
        )
        self.assertEqual(
            route_after_rewrite(
                {
                    "questionIsClear": True,
                    "intent_type": "general_retrieval",
                    "task_intent": "metadata_query",
                    "rewrittenQuestions": ["How many papers?"],
                }
            ),
            "metadata_query_response",
        )
        self.assertEqual(
            route_after_rewrite(
                {
                    "questionIsClear": True,
                    "intent_type": "single_doc_close_reading",
                    "task_intent": "single_paper_summary",
                    "referenced_documents": ["paper-a"],
                    "rewrittenQuestions": ["Summarize this paper."],
                }
            ),
            "single_paper_summary",
        )

    def test_intent_analysis_accepts_fine_grained_fields_and_old_payloads(self):
        old_payload = IntentAnalysis.model_validate(
            {
                "intent_type": "general_retrieval",
                "resolved_query": "What changed?",
                "needs_clarification": False,
            }
        )
        self.assertEqual(old_payload.task_intent, "single_paper_qa")

        payload = IntentAnalysis.model_validate(
            {
                "intent_type": "single_doc_close_reading",
                "task_intent": "method_explanation",
                "resolved_query": "Explain the method pipeline.",
                "target_papers": ["paper-a"],
                "retrieval_scope": ["method", "algorithm"],
                "answer_format": "structured_explanation",
                "need_memory": False,
                "need_metadata_filter": False,
                "confidence": 0.86,
                "needs_clarification": False,
            }
        )
        self.assertEqual(payload.task_intent, "method_explanation")
        self.assertEqual(payload.target_papers, ["paper-a"])
        self.assertEqual(payload.answer_format, "structured_explanation")

    def test_build_retrieval_plan_prefers_per_paper_compare_defaults(self):
        plan = _build_retrieval_plan(
            query="Compare the performance of paper A and paper B",
            intent_type="cross_doc_comparison",
            referenced_documents=["paper-a", "paper-b"],
            comparison_targets=["paper-a", "paper-b"],
            workspace_context={},
            paper_catalog=[],
        )

        self.assertEqual(plan["mode"], "per_paper_compare")
        self.assertEqual(plan["paper_ids"], ["paper-a", "paper-b"])
        self.assertIn("performance", plan["section_hints"])
        self.assertIn("table", plan["content_type_hints"])
        self.assertEqual(plan["per_paper_limit"], 3)

    def test_build_retrieval_plan_merges_partial_llm_targets_for_comparison(self):
        plan = _build_retrieval_plan(
            query="Compare the main contributions of the CEDC-ABE paper and the PM-ABE paper.",
            intent_type="cross_doc_comparison",
            referenced_documents=["cedc", "pmabe"],
            comparison_targets=["cedc", "pmabe"],
            workspace_context={},
            paper_catalog=[],
            proposed_plan={"paper_ids": ["pmabe"]},
        )

        self.assertEqual(plan["paper_ids"], ["cedc", "pmabe"])

    def test_build_retrieval_plan_does_not_override_explicit_reference_with_selected_focus(self):
        plan = _build_retrieval_plan(
            query="What is the CEDC-ABE scheme?",
            intent_type="single_doc_close_reading",
            referenced_documents=["paper-cedc"],
            comparison_targets=["paper-cedc"],
            workspace_context={"focus_paper_id": "paper-pmabe"},
            paper_catalog=[],
        )

        self.assertEqual(plan["paper_ids"], ["paper-cedc"])

    def test_build_retrieval_plan_tunes_single_paper_summary(self):
        plan = _build_retrieval_plan(
            query="Summarize this paper.",
            intent_type="single_doc_close_reading",
            task_intent="single_paper_summary",
            referenced_documents=["paper-a"],
            comparison_targets=[],
            workspace_context={},
            paper_catalog=[],
        )

        self.assertEqual(plan["mode"], "single_doc")
        self.assertEqual(plan["answer_format"], "structured_summary")
        self.assertIn("abstract", plan["retrieval_scope"])
        self.assertIn("limitation", plan["retrieval_scope"])
        self.assertGreaterEqual(plan["per_paper_limit"], 8)

    def test_build_retrieval_plan_tunes_method_explanation(self):
        plan = _build_retrieval_plan(
            query="Explain the pipeline and loss function.",
            intent_type="single_doc_close_reading",
            task_intent="method_explanation",
            referenced_documents=["paper-a"],
            comparison_targets=[],
            workspace_context={},
            paper_catalog=[],
        )

        self.assertEqual(plan["answer_format"], "structured_explanation")
        self.assertIn("method", plan["section_hints"])
        self.assertIn("formula", plan["section_hints"])
        self.assertIn("training objective", plan["retrieval_scope"])

    def test_build_retrieval_plan_tunes_citation_finding_and_metadata_query(self):
        citation_plan = _build_retrieval_plan(
            query="Which paper supports this claim?",
            intent_type="general_retrieval",
            task_intent="citation_finding",
            referenced_documents=[],
            comparison_targets=[],
            workspace_context={},
            paper_catalog=[],
        )
        self.assertEqual(citation_plan["answer_format"], "evidence_list")

        metadata_plan = _build_retrieval_plan(
            query="How many papers are in the workspace?",
            intent_type="general_retrieval",
            task_intent="metadata_query",
            referenced_documents=[],
            comparison_targets=[],
            workspace_context={},
            paper_catalog=[],
        )
        self.assertTrue(metadata_plan["need_metadata_filter"])
        self.assertEqual(metadata_plan["answer_format"], "metadata_list")

    def test_apply_artifact_kind_builds_comparison_table(self):
        payload, rendered = _apply_artifact_kind(
            {
                "artifact_kind": "build_comparison_table",
                "originalQuery": "Build a comparison table for these papers",
            },
            "unused",
            profiles=[
                {
                    "paper_id": "paper-a",
                    "title": "Paper A",
                    "problem": "Fine-grained access control",
                    "core_method": "Lattice-based construction",
                    "assumptions": ["RLWE", "honest authority"],
                },
                {
                    "paper_id": "paper-b",
                    "title": "Paper B",
                    "problem": "Blockchain auditing",
                    "core_method": "ABE with audit log",
                    "assumptions": ["Bilinear groups"],
                },
            ],
        )

        self.assertEqual(payload["kind"], "build_comparison_table")
        self.assertEqual(len(payload["rows"]), 2)
        self.assertIn("| Paper | Problem | Core Method | Assumptions |", rendered)
        self.assertIn("| paper-a | Fine-grained access control | Lattice-based construction | RLWE, honest authority |", rendered)

    def test_apply_artifact_kind_builds_notes_from_evidence(self):
        payload, rendered = _apply_artifact_kind(
            {
                "artifact_kind": "export_notes",
                "originalQuery": "整理这篇论文的笔记",
            },
            "The paper focuses on bilateral access control.",
            evidence=[
                {"section": "System Model", "content": "The system model introduces users, edge servers, and cloud."},
                {"section": "Construction", "content": "The construction defines Setup, KeyGen, and Encrypt."},
            ],
        )

        self.assertEqual(payload["kind"], "export_notes")
        self.assertEqual(len(payload["sections"]), 2)
        self.assertIn("# Research Notes", rendered)
        self.assertIn("## System Model", rendered)
        self.assertIn("## Takeaway", rendered)

    def test_close_reading_prefers_resolved_reference_over_selected_focus(self):
        class _FakeRetriever:
            def __init__(self):
                self.paper_ids = []

            def search(self, query, workspace_id, paper_id=None, limit=4, query_profile=None, query_variants=None):
                self.paper_ids.append(paper_id)
                doc = SimpleNamespace(
                    metadata={
                        "parent_id": "parent-1",
                        "paper_id": paper_id,
                        "section": "Construction",
                        "page_start": 1,
                        "page_end": 1,
                        "source": f"{paper_id}.pdf",
                        "content_type": "paragraph",
                    },
                    page_content="The scheme uses cloud-edge-device collaboration.",
                )
                return {
                    "child_hits": [doc],
                    "parent_chunks": [
                        {
                            "parent_id": "parent-1",
                            "metadata": {
                                "paper_id": paper_id,
                                "section": "Construction",
                                "page_start": 1,
                                "page_end": 1,
                                "source": f"{paper_id}.pdf",
                                "content_type": "paragraph",
                            },
                            "content": "The scheme uses cloud-edge-device collaboration.",
                        }
                    ],
                    "rerank_backend": "heuristic_fallback",
                }

        class _FakeLlm:
            def invoke(self, messages):
                return SimpleNamespace(content="Resolved answer.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-cedc", "source_name": "paper-cedc.pdf"})

            retriever = _FakeRetriever()
            result = close_reading(
                {
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "paper-pmabe",
                    },
                    "working_memory": {
                        "current_dialogue_paper_id": "paper-cedc",
                    },
                    "retrieval_plan": {"paper_ids": ["paper-cedc"]},
                    "referenced_documents": ["paper-cedc"],
                    "rewrittenQuestions": ["What is the CEDC-ABE scheme?"],
                    "originalQuery": "What is the CEDC-ABE scheme?",
                    "intent_type": "single_doc_close_reading",
                    "artifact_kind": "",
                    "verification_status": "",
                },
                _FakeLlm(),
                retriever,
                store,
                default_workspace_id="demo",
            )

        self.assertEqual(retriever.paper_ids, ["paper-cedc"])
        self.assertEqual(result["retrieved_paper_ids"], ["paper-cedc"])
        self.assertIn("Resolved answer.", result["final_answer"])

    def test_orchestrator_uses_list_workspace_papers_first_for_inventory_query(self):
        class _FakeLlmWithTools:
            def __init__(self):
                self.messages = None

            def invoke(self, messages):
                self.messages = messages
                return AIMessage(content="", tool_calls=[])

        fake_llm = _FakeLlmWithTools()
        orchestrator(
            {
                "question": "What papers are available in the current workspace?",
                "messages": [],
                "workspace_context": {"workspace_id": "demo"},
                "context_summary": "",
            },
            fake_llm,
        )

        self.assertIsNotNone(fake_llm.messages)
        self.assertIn("list_workspace_papers", fake_llm.messages[-1].content)

    def test_workspace_inventory_response_lists_titles_and_paper_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-a", "title": "Paper A", "source_name": "Paper A.pdf"})
            store.register_paper("demo", {"paper_id": "paper-b", "title": "Paper B", "source_name": "Paper B.pdf"})

            result = workspace_inventory_response(
                {
                    "workspace_context": {"workspace_id": "demo"},
                    "originalQuery": "What papers are available in the current workspace?",
                    "intent_type": "general_retrieval",
                    "artifact_kind": "",
                    "verification_status": "",
                },
                store,
                default_workspace_id="demo",
            )

        self.assertEqual(result["retrieved_paper_ids"], ["paper-a", "paper-b"])
        self.assertIn("paper-a", result["final_answer"])
        self.assertIn("Paper B", result["final_answer"])

    def test_metadata_query_response_counts_and_filters_catalog(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-a", "title": "Paper A", "source_name": "A.pdf", "year": 2024, "tags": ["rag"], "read_status": "read"})
            store.register_paper("demo", {"paper_id": "paper-b", "title": "Paper B", "source_name": "B.pdf", "year": 2025, "tags": ["agent"], "read_status": "unread"})

            result = metadata_query_response(
                {
                    "workspace_context": {"workspace_id": "demo"},
                    "originalQuery": "How many papers are in the current workspace?",
                    "intent_type": "general_retrieval",
                    "task_intent": "metadata_query",
                },
                store,
                default_workspace_id="demo",
            )
            self.assertIn("2 paper", result["final_answer"])

            filtered = metadata_query_response(
                {
                    "workspace_context": {"workspace_id": "demo"},
                    "originalQuery": "List unread papers",
                    "intent_type": "general_retrieval",
                    "task_intent": "metadata_query",
                },
                store,
                default_workspace_id="demo",
            )
            self.assertIn("paper-b", filtered["final_answer"])
            self.assertNotIn("paper-a: Paper A", filtered["final_answer"])

    def test_metadata_query_response_reports_read_only_write_requests(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-a", "title": "Paper A", "source_name": "A.pdf"})

            result = metadata_query_response(
                {
                    "workspace_context": {"workspace_id": "demo"},
                    "originalQuery": "Add tag important to paper-a",
                    "intent_type": "general_retrieval",
                    "task_intent": "metadata_query",
                },
                store,
                default_workspace_id="demo",
            )

            self.assertIn("read-only", result["final_answer"])

    def test_workspace_memory_qa_uses_memory_evidence(self):
        class _FakeLlm:
            def invoke(self, messages):
                content = messages[-1].content
                self.last_content = content
                return SimpleNamespace(content="Memory answer based on previous interaction.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.save_working_memory_snapshot("demo", {"current_research_question": "reranking evidence"})
            store.record_interaction("demo", {"query": "Summarize reranking evidence", "intent_type": "general_retrieval", "paper_ids": ["paper-a"]})
            llm = _FakeLlm()

            result = workspace_memory_qa(
                {
                    "workspace_context": {"workspace_id": "demo"},
                    "working_memory": {"recent_papers": ["paper-a"]},
                    "memory_context": {},
                    "memory_hits": [],
                    "originalQuery": "What did I summarize last time about reranking?",
                    "intent_type": "general_retrieval",
                    "task_intent": "workspace_memory_qa",
                },
                llm,
                store,
                default_workspace_id="demo",
            )

            self.assertIn("Memory answer", result["final_answer"])
            self.assertIn("Memory evidence", llm.last_content)

    def test_workspace_memory_qa_reports_insufficient_memory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)

            result = workspace_memory_qa(
                {
                    "workspace_context": {"workspace_id": "demo"},
                    "working_memory": {},
                    "memory_context": {},
                    "memory_hits": [],
                    "originalQuery": "What did I read last time?",
                    "intent_type": "general_retrieval",
                    "task_intent": "workspace_memory_qa",
                },
                SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="")),
                store,
                default_workspace_id="demo",
            )

            self.assertIn("workspace memory", result["final_answer"])

    def test_single_paper_summary_aggregates_parent_sections(self):
        class _FakeParentStore:
            def load_paper(self, workspace_id, paper_id):
                return [
                    {"parent_id": "p0", "metadata": {"paper_id": paper_id, "section": "Abstract", "page_start": 1, "page_end": 1, "source": "Paper.pdf"}, "content": "We propose the main contribution."},
                    {"parent_id": "p1", "metadata": {"paper_id": paper_id, "section": "Method", "page_start": 2, "page_end": 3, "source": "Paper.pdf"}, "content": "The method has three modules."},
                    {"parent_id": "p2", "metadata": {"paper_id": paper_id, "section": "Experimental Results", "page_start": 4, "page_end": 5, "source": "Paper.pdf"}, "content": "Experiments improve accuracy."},
                    {"parent_id": "p3", "metadata": {"paper_id": paper_id, "section": "Conclusion", "page_start": 6, "page_end": 6, "source": "Paper.pdf"}, "content": "The paper concludes the approach is effective."},
                ]

        class _FakeLlm:
            def invoke(self, messages):
                self.last_content = messages[-1].content
                return SimpleNamespace(content="Structured summary with limitation missing.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-a", "title": "Paper A", "source_name": "Paper.pdf"})
            llm = _FakeLlm()

            result = single_paper_summary(
                {
                    "workspace_context": {"workspace_id": "demo", "focus_paper_id": "paper-a"},
                    "retrieval_plan": {"paper_ids": ["paper-a"]},
                    "referenced_documents": ["paper-a"],
                    "originalQuery": "Summarize this paper.",
                    "intent_type": "single_doc_close_reading",
                    "task_intent": "single_paper_summary",
                },
                llm,
                _FakeParentStore(),
                store,
                default_workspace_id="demo",
            )

            self.assertIn("Structured summary", result["final_answer"])
            self.assertIn("## limitation", llm.last_content)
            self.assertIn("Abstract", llm.last_content)
            self.assertEqual(result["retrieved_paper_ids"], ["paper-a"])


if __name__ == "__main__":
    unittest.main()
