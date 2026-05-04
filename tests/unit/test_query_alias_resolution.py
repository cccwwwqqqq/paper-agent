import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

from agentic_rag.agents.nodes import _paper_aliases, _resolve_query_aliases, rewrite_query
from agentic_rag.agents.schemas import IntentAnalysis
from agentic_rag.settings import load_settings
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore


class _FakeStructuredLlm:
    def __init__(self, payload):
        self.payload = payload

    def with_config(self, temperature=0.1):
        return self

    def with_structured_output(self, schema, method="function_calling"):
        return self

    def invoke(self, messages):
        return IntentAnalysis.model_validate(self.payload)


class QueryAliasResolutionTests(unittest.TestCase):
    def test_extracts_aliases_from_markdown_and_matches_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            markdown_path = Path(tmp_dir) / "paper.md"
            markdown_path.write_text(
                "## Privacy-Preserving Attribute-Based Bilateral Access Control with Zero-Knowledge Pre-Verification for IIoT\n\n"
                "We propose a Bandwidth-Efficient and Privacy-Preserving Bilateral Access Control (BE-PBAC) scheme.\n",
                encoding="utf-8",
            )
            paper = {
                "paper_id": "privacy-preserving-attribute-based-bilateral-acc-f71f114d",
                "title": "Privacy-Preserving Attribute-Based Bilateral Access Control with Zero-Knowledge Pre-Verification for IIoT",
                "source_name": "Privacy-Preserving Attribute-Based Bilateral Access Control with Zero-Knowledge Pre-Verification for IIoT.pdf",
                "markdown_path": str(markdown_path),
            }

            aliases = _paper_aliases(paper)
            matches = _resolve_query_aliases("BE-PBAC这篇论文讲了什么", [paper])

            self.assertIn("be-pbac", aliases)
            self.assertEqual(matches, ["privacy-preserving-attribute-based-bilateral-acc-f71f114d"])

    def test_exact_hyphenated_alias_matching_avoids_generic_abe_false_positive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cedc_path = Path(tmp_dir) / "cedc.md"
            cedc_path.write_text(
                "We call the scheme CEDC-ABE throughout the paper.\n",
                encoding="utf-8",
            )
            pmabe_path = Path(tmp_dir) / "pmabe.md"
            pmabe_path.write_text(
                "We call the scheme PM-ABE throughout the paper.\n",
                encoding="utf-8",
            )

            papers = [
                {
                    "paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "title": "Document Overview",
                    "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme.pdf",
                    "markdown_path": str(cedc_path),
                },
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                    "markdown_path": str(pmabe_path),
                },
            ]

            matches = _resolve_query_aliases("In the CEDC-ABE paper, what is blockchain used for?", papers)

            self.assertEqual(matches, ["a-cloud-edge-device-collaborative-attribute-base-499d8e10"])

    def test_derives_cedc_abe_alias_from_source_name_when_markdown_lacks_it(self):
        paper = {
            "paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
            "title": "Document Overview",
            "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme for Industrial Internet with a Semi-Trusted Environment.pdf",
            "markdown_path": "",
        }

        aliases = _paper_aliases(paper)
        matches = _resolve_query_aliases("What is the CEDC-ABE scheme?", [paper])

        self.assertIn("cedc-abe", aliases)
        self.assertNotIn("cp-abe", aliases)
        self.assertEqual(matches, ["a-cloud-edge-device-collaborative-attribute-base-499d8e10"])

    def test_rewrite_query_does_not_reuse_focus_paper_for_unknown_explicit_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )
            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="What is the XYZ-ABE scheme?")],
                    "workspace_context": {"workspace_id": "demo", "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa"},
                    "working_memory": {"focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa"},
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "What is the XYZ-ABE scheme?",
                        "rewritten_questions": ["What is the XYZ-ABE scheme?"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertFalse(result["questionIsClear"])
            self.assertIn("xyz-abe", result["clarification_question"].lower())
            self.assertNotEqual(
                result["workspace_context"].get("focus_paper_id"),
                "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
            )

    def test_rewrite_query_keeps_current_focus_paper_in_comparison_follow_up(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "title": "Document Overview",
                    "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme for Industrial Internet with a Semi-Trusted Environment.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="What is the biggest difference between it and PM-ABE?")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    },
                    "working_memory": {
                        "focus_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                        "recent_papers": ["a-cloud-edge-device-collaborative-attribute-base-499d8e10"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "cross_doc_comparison",
                        "resolved_query": "What is the biggest difference between it and PM-ABE?",
                        "rewritten_questions": ["What is the biggest difference between it and PM-ABE?"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(
                result["referenced_documents"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )
            self.assertEqual(
                result["comparison_targets"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )

    def test_rewrite_query_prefers_working_memory_focus_over_selected_dropdown_focus(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "title": "Document Overview",
                    "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme for Industrial Internet with a Semi-Trusted Environment.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="What is the biggest difference between it and PM-ABE?")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "selected_focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    },
                    "working_memory": {
                        "focus_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                        "recent_papers": ["a-cloud-edge-device-collaborative-attribute-base-499d8e10"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "cross_doc_comparison",
                        "resolved_query": "What is the biggest difference between it and PM-ABE?",
                        "rewritten_questions": ["What is the biggest difference between it and PM-ABE?"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertEqual(
                result["referenced_documents"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )
            self.assertEqual(
                result["retrieval_plan"]["paper_ids"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )

    def test_rewrite_query_prefers_current_dialogue_paper_over_selected_focus_for_pronoun_follow_up(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "title": "Document Overview",
                    "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme for Industrial Internet with a Semi-Trusted Environment.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="What is the biggest difference between it and PM-ABE?")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "selected_focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    },
                    "working_memory": {
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "current_dialogue_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                        "recent_papers": ["a-cloud-edge-device-collaborative-attribute-base-499d8e10"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "cross_doc_comparison",
                        "resolved_query": "What is the biggest difference between it and PM-ABE?",
                        "rewritten_questions": ["What is the biggest difference between it and PM-ABE?"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(
                result["referenced_documents"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )
            self.assertEqual(
                result["retrieval_plan"]["paper_ids"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )
            self.assertEqual(
                result["working_memory"]["current_dialogue_paper_id"],
                "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
            )

    def test_rewrite_query_detects_chinese_difference_question_as_comparison(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "title": "Document Overview",
                    "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme for Industrial Internet with a Semi-Trusted Environment.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="那它和PM-ABE最大的区别是什么")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "selected_focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    },
                    "working_memory": {
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "current_dialogue_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                        "recent_papers": ["a-cloud-edge-device-collaborative-attribute-base-499d8e10"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "那它和PM-ABE最大的区别是什么",
                        "rewritten_questions": ["那它和PM-ABE最大的区别是什么"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(result["intent_type"], "cross_doc_comparison")
            self.assertEqual(
                result["retrieval_plan"]["paper_ids"],
                [
                    "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                    "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                ],
            )

    def test_rewrite_query_treats_current_workspace_papers_summary_as_literature_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "cedc",
                    "title": "CEDC",
                    "source_name": "CEDC.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pmabe",
                    "title": "PM-ABE",
                    "source_name": "PM-ABE.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="总结当前工作区这些论文的研究主题")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "cedc",
                        "selected_focus_paper_id": "cedc",
                    },
                    "working_memory": {
                        "focus_paper_id": "cedc",
                        "current_dialogue_paper_id": "cedc",
                        "recent_papers": ["cedc"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "single_doc_close_reading",
                        "resolved_query": "总结当前工作区这些论文的研究主题",
                        "rewritten_questions": ["总结当前工作区这些论文的研究主题"],
                        "referenced_documents": ["cedc"],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(result["intent_type"], "literature_review")
            self.assertEqual(result["referenced_documents"], [])
            self.assertEqual(result["comparison_targets"], ["cedc", "pmabe"])
            self.assertEqual(result["retrieval_plan"]["mode"], "literature_review")
            self.assertEqual(result["retrieval_plan"]["paper_ids"], ["cedc", "pmabe"])
            self.assertIsNone(result["workspace_context"]["focus_paper_id"])

    def test_rewrite_query_treats_workspace_inventory_question_as_general_retrieval_and_clears_focus(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "cedc",
                    "title": "CEDC",
                    "source_name": "CEDC.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pmabe",
                    "title": "PM-ABE",
                    "source_name": "PM-ABE.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="What papers are available in the current workspace?")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "cedc",
                        "selected_focus_paper_id": "cedc",
                    },
                    "working_memory": {
                        "focus_paper_id": "cedc",
                        "current_dialogue_paper_id": "cedc",
                        "recent_papers": ["cedc"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "single_doc_close_reading",
                        "resolved_query": "What papers are available in the current workspace?",
                        "rewritten_questions": ["What papers are available in the current workspace?"],
                        "referenced_documents": ["cedc"],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(result["intent_type"], "general_retrieval")
            self.assertEqual(result["referenced_documents"], [])
            self.assertEqual(result["retrieval_plan"]["mode"], "general")
            self.assertIsNone(result["workspace_context"]["focus_paper_id"])
            self.assertTrue(result["workspace_inventory_query"])

    def test_rewrite_query_does_not_treat_cifar10_as_missing_paper_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "cedc",
                    "title": "CEDC",
                    "source_name": "CEDC.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="Which papers in the current workspace report CIFAR-10 accuracy?")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "cedc",
                    },
                    "working_memory": {
                        "focus_paper_id": "cedc",
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "Which papers in the current workspace report CIFAR-10 accuracy?",
                        "rewritten_questions": ["Which papers in the current workspace report CIFAR-10 accuracy?"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(result["intent_type"], "general_retrieval")
            self.assertEqual(result["clarification_question"], "")

    def test_rewrite_query_resolves_previous_paper_follow_up_and_carries_question(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "paper-a",
                    "title": "Paper A",
                    "source_name": "Paper A.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "paper-b",
                    "title": "Paper B",
                    "source_name": "Paper B.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="上一篇呢")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "paper-b",
                        "selected_focus_paper_id": "paper-b",
                        "intent_type": "single_doc_close_reading",
                    },
                    "working_memory": {
                        "focus_paper_id": "paper-b",
                        "current_dialogue_paper_id": "paper-b",
                        "current_research_question": "这篇论文基于什么假设",
                        "recent_papers": ["paper-b", "paper-a"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "上一篇呢",
                        "rewritten_questions": ["上一篇呢"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(result["intent_type"], "single_doc_close_reading")
            self.assertEqual(result["referenced_documents"], ["paper-a"])
            self.assertEqual(result["rewrittenQuestions"], ["这篇论文基于什么假设"])
            self.assertEqual(result["working_memory"]["current_dialogue_paper_id"], "paper-a")
            self.assertEqual(result["working_memory"]["current_research_question"], "这篇论文基于什么假设")
            self.assertEqual(result["working_memory"]["recent_papers"][:2], ["paper-a", "paper-b"])

    def test_rewrite_query_stores_single_paper_question_as_relative_template(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "be-pbac",
                    "title": "BE-PBAC",
                    "source_name": "Privacy-Preserving Attribute-Based Bilateral Access Control with Zero-Knowledge Pre-Verification for IIoT.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="BE-PBAC基于什么假设")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": None,
                        "selected_focus_paper_id": None,
                        "intent_type": "general_retrieval",
                    },
                    "working_memory": {},
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "BE-PBAC基于什么假设",
                        "rewritten_questions": ["BE-PBAC基于什么假设"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertEqual(result["intent_type"], "single_doc_close_reading")
            self.assertEqual(result["referenced_documents"], ["be-pbac"])
            self.assertEqual(result["working_memory"]["current_research_question"], "这篇论文基于什么假设")

    def test_rewrite_query_previous_paper_follow_up_after_explicit_alias_question_uses_relative_template(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe",
                    "title": "PM-ABE",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "be-pbac",
                    "title": "BE-PBAC",
                    "source_name": "Privacy-Preserving Attribute-Based Bilateral Access Control with Zero-Knowledge Pre-Verification for IIoT.pdf",
                },
            )

            first = rewrite_query(
                {
                    "messages": [HumanMessage(content="PM-ABE讲了什么")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": None,
                        "selected_focus_paper_id": None,
                        "intent_type": "general_retrieval",
                    },
                    "working_memory": {},
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "PM-ABE讲了什么",
                        "rewritten_questions": ["PM-ABE讲了什么"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            second = rewrite_query(
                {
                    "messages": [HumanMessage(content="BE-PBAC基于什么假设")],
                    "workspace_context": first["workspace_context"],
                    "working_memory": first["working_memory"],
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "BE-PBAC基于什么假设",
                        "rewritten_questions": ["BE-PBAC基于什么假设"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            third = rewrite_query(
                {
                    "messages": [HumanMessage(content="上一篇呢")],
                    "workspace_context": second["workspace_context"],
                    "working_memory": second["working_memory"],
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "general_retrieval",
                        "resolved_query": "上一篇呢",
                        "rewritten_questions": ["上一篇呢"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertEqual(second["working_memory"]["current_research_question"], "这篇论文基于什么假设")
            self.assertEqual(third["intent_type"], "single_doc_close_reading")
            self.assertEqual(third["referenced_documents"], ["pm-abe"])
            self.assertEqual(third["rewrittenQuestions"], ["这篇论文基于什么假设"])

    def test_rewrite_query_clarifies_when_comparison_still_only_has_one_paper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="Compare PM-ABE with it")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "selected_focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    },
                    "working_memory": {
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "recent_papers": ["pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "cross_doc_comparison",
                        "resolved_query": "Compare PM-ABE with it",
                        "rewritten_questions": ["Compare PM-ABE with it"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertFalse(result["questionIsClear"])
            self.assertIn("comparison question", result["clarification_question"].lower())

    def test_rewrite_query_merges_alias_matches_with_partial_llm_comparison_plan(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "cedc",
                    "title": "Document Overview",
                    "source_name": "A Cloud-Edge-Device Collaborative Attribute-Based Encryption Data Sharing Scheme for Industrial Internet with a Semi-Trusted Environment.pdf",
                },
            )
            store.register_paper(
                "demo",
                {
                    "paper_id": "pmabe",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="Compare the main contributions of the CEDC-ABE paper and the PM-ABE paper.")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "cedc",
                        "selected_focus_paper_id": "cedc",
                    },
                    "working_memory": {
                        "focus_paper_id": "cedc",
                        "recent_papers": ["cedc"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "cross_doc_comparison",
                        "resolved_query": "Compare the main contributions of the CEDC-ABE paper and the PM-ABE paper.",
                        "rewritten_questions": ["Compare the main contributions of the CEDC-ABE paper and the PM-ABE paper."],
                        "referenced_documents": [],
                        "comparison_targets": ["pmabe"],
                        "retrieval_plan": {"paper_ids": ["pmabe"]},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertTrue(result["questionIsClear"])
            self.assertEqual(result["intent_type"], "cross_doc_comparison")
            self.assertEqual(result["comparison_targets"], ["cedc", "pmabe"])
            self.assertEqual(result["retrieval_plan"]["paper_ids"], ["cedc", "pmabe"])

    def test_rewrite_query_clarifies_when_pronoun_and_alias_collapse_to_same_paper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper(
                "demo",
                {
                    "paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    "title": "PM-ABE: Puncturable Bilateral Fine-Grained Access Control From Lattices for Secret Sharing",
                    "source_name": "PM-ABE_Puncturable_Bilateral_Fine-Grained_Access_Control_From_Lattices_for_Secret_Sharing.pdf",
                },
            )

            result = rewrite_query(
                {
                    "messages": [HumanMessage(content="What is the biggest difference between it and PM-ABE?")],
                    "workspace_context": {
                        "workspace_id": "demo",
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                    },
                    "working_memory": {
                        "focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                        "recent_papers": ["pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa"],
                    },
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "cross_doc_comparison",
                        "resolved_query": "What is the biggest difference between it and PM-ABE?",
                        "rewritten_questions": ["What is the biggest difference between it and PM-ABE?"],
                        "referenced_documents": [],
                        "comparison_targets": [],
                        "retrieval_plan": {},
                        "artifact_kind": "",
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                ),
                store,
                default_workspace_id="demo",
            )

            self.assertFalse(result["questionIsClear"])
            self.assertIn("comparison question", result["clarification_question"].lower())

    def test_rewrite_query_sets_task_intent_for_single_paper_summary_and_method_explanation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-a", "title": "Paper A", "source_name": "Paper A.pdf"})

            summary = rewrite_query(
                {
                    "messages": [HumanMessage(content="\u603b\u7ed3\u8fd9\u7bc7\u8bba\u6587")],
                    "workspace_context": {"workspace_id": "demo", "focus_paper_id": "paper-a"},
                    "working_memory": {"focus_paper_id": "paper-a", "current_dialogue_paper_id": "paper-a"},
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "single_doc_close_reading",
                        "resolved_query": "\u603b\u7ed3\u8fd9\u7bc7\u8bba\u6587",
                        "needs_clarification": False,
                    }
                ),
                store,
                default_workspace_id="demo",
            )
            self.assertEqual(summary["intent_type"], "single_doc_close_reading")
            self.assertEqual(summary["task_intent"], "single_paper_summary")
            self.assertEqual(summary["retrieval_plan"]["answer_format"], "structured_summary")

            method = rewrite_query(
                {
                    "messages": [HumanMessage(content="How does this pipeline and loss function work?")],
                    "workspace_context": {"workspace_id": "demo", "focus_paper_id": "paper-a"},
                    "working_memory": {"focus_paper_id": "paper-a", "current_dialogue_paper_id": "paper-a"},
                    "conversation_summary": "",
                },
                _FakeStructuredLlm(
                    {
                        "intent_type": "single_doc_close_reading",
                        "resolved_query": "How does this pipeline and loss function work?",
                        "needs_clarification": False,
                    }
                ),
                store,
                default_workspace_id="demo",
            )
            self.assertEqual(method["task_intent"], "method_explanation")
            self.assertIn("formula", method["retrieval_plan"]["section_hints"])

    def test_rewrite_query_sets_task_intent_for_citation_memory_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-a", "title": "Paper A", "source_name": "Paper A.pdf"})

            def run_query(query):
                return rewrite_query(
                    {
                        "messages": [HumanMessage(content=query)],
                        "workspace_context": {"workspace_id": "demo"},
                        "working_memory": {},
                        "conversation_summary": "",
                    },
                    _FakeStructuredLlm(
                        {
                            "intent_type": "general_retrieval",
                            "resolved_query": query,
                            "rewritten_questions": [query],
                            "referenced_documents": [],
                            "comparison_targets": [],
                            "retrieval_plan": {},
                            "artifact_kind": "",
                            "needs_clarification": False,
                            "clarification_question": "",
                        }
                    ),
                    store,
                    default_workspace_id="demo",
                )

            citation = run_query("Which paper supports this claim about reranking?")
            self.assertEqual(citation["task_intent"], "citation_finding")
            self.assertEqual(citation["retrieval_plan"]["answer_format"], "evidence_list")

            memory = run_query("What did I summarize last time about this direction?")
            self.assertEqual(memory["task_intent"], "workspace_memory_qa")
            self.assertTrue(memory["retrieval_plan"]["need_memory"])

            metadata = run_query("How many papers are in the current workspace?")
            self.assertEqual(metadata["task_intent"], "metadata_query")
            self.assertTrue(metadata["retrieval_plan"]["need_metadata_filter"])
            self.assertTrue(metadata["workspace_inventory_query"])


if __name__ == "__main__":
    unittest.main()
