import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag.document_chunker import DocumentChuncker
from agentic_rag.parsers import create_pdf_parser
from agentic_rag.services.ingestion_service import IngestionService
from agentic_rag.settings import load_settings
from agentic_rag.storage.parent_store import ParentStoreManager
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore
from agentic_rag.utils.ids import resolve_paper_id
from eval import evaluate as eval_module


class FakeCollection:
    def __init__(self):
        self.added_documents = []

    def add_documents(self, documents):
        self.added_documents.extend(documents)


class FakeVectorDb:
    def __init__(self):
        self.collection = FakeCollection()
        self.deleted_workspaces = []

    def get_collection(self, collection_name):
        return self.collection

    def delete_workspace_points(self, collection_name, workspace_id):
        self.deleted_workspaces.append((collection_name, workspace_id))


class FakeGraph:
    def __init__(self):
        self.values = {}
        self.next = False

    def get_state(self, config):
        return SimpleNamespace(next=self.next, values=self.values)

    def update_state(self, config, state_update):
        self.values["workspace_context"] = state_update.get("workspace_context", {})

    def invoke(self, invoke_input, config=None):
        state_update = invoke_input or {}
        workspace_context = state_update.get("workspace_context", self.values.get("workspace_context", {}))
        self.values = {
            "intent_type": "general_retrieval",
            "final_answer": "Supported answer from the fake runtime.",
            "clarification_question": "",
            "workspace_context": workspace_context,
            "referenced_documents": ["paper-1"],
            "comparison_targets": [],
            "retrieved_chunks": ["chunk-1"],
            "retrieved_parent_chunks": ["paper-1_parent_0"],
            "retrieved_paper_ids": ["paper-1"],
            "retrieved_sections": ["Introduction"],
            "paper_profiles": [{"paper_id": "paper-1", "title": "Paper One"}],
            "messages": [AIMessage(content="Supported answer from the fake runtime.")],
        }


class FakeWorkspaceMemory:
    def load_working_memory_snapshot(self, workspace_id):
        return {}


class FakeRagSystem:
    def __init__(self):
        self.agent_graph = FakeGraph()
        self.workspace_memory = FakeWorkspaceMemory()
        self.active_context = {"workspace_context": {"workspace_id": "default", "focus_paper_id": None}}
        self.reset_calls = 0

    def reset_thread(self):
        self.reset_calls += 1
        self.agent_graph.values = {}
        self.agent_graph.next = False

    def set_workspace_context(self, workspace_id, focus_paper_id=None, intent_type="general_retrieval"):
        self.active_context = {
            "workspace_context": {
                "workspace_id": workspace_id,
                "focus_paper_id": focus_paper_id,
                "intent_type": intent_type,
            }
        }

    def get_workspace_context(self):
        return self.active_context

    def get_config(self):
        return {"configurable": {"thread_id": "fake-thread"}}


class IngestionAndEvalIntegrationTests(unittest.TestCase):
    def test_ingestion_service_indexes_markdown_and_clears_workspace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(os.environ, {}, clear=True):
                root = Path(tmp_dir)
                settings = load_settings(root_dir=root, use_env=False)
                parser = create_pdf_parser(settings)
                chunker = DocumentChuncker(settings)
                parent_store = ParentStoreManager(settings)
                workspace_memory = WorkspaceMemoryStore(settings)
                vector_db = FakeVectorDb()
                source_doc = root / "sample.md"
                source_doc.write_text("# Introduction\n\nThis is a test document.\n", encoding="utf-8")

                rag_system = SimpleNamespace(
                    parser=parser,
                    chunker=chunker,
                    parent_store=parent_store,
                    workspace_memory=workspace_memory,
                    vector_db=vector_db,
                    collection_name=settings.child_collection,
                    settings=settings,
                )
                service = IngestionService(rag_system, settings)

                added, skipped = service.add_documents([str(source_doc)], workspace_id="demo")

                self.assertEqual((added, skipped), (1, 0))
                self.assertTrue(vector_db.collection.added_documents)
                stored_papers = workspace_memory.list_papers("demo")
                self.assertEqual(len(stored_papers), 1)
                expected_paper_id = resolve_paper_id("sample")
                markdown_path = settings.markdown_dir / "demo" / f"{expected_paper_id}.md"
                self.assertTrue(markdown_path.exists())
                source_path = settings.source_docs_dir / "demo" / f"{expected_paper_id}.md"
                self.assertTrue(source_path.exists())
                self.assertEqual(stored_papers[0]["source_path"], str(source_path))
                parent_chunks = parent_store.load_paper("demo", expected_paper_id)
                self.assertTrue(parent_chunks)
                self.assertEqual(service.get_markdown_files("demo"), ["sample.md"])

                added_again, skipped_again = service.add_documents([str(source_doc)], workspace_id="demo")
                self.assertEqual((added_again, skipped_again), (0, 1))

                service.clear_all("demo")

                self.assertEqual(workspace_memory.list_papers("demo"), [])
                self.assertEqual(parent_store.load_paper("demo", expected_paper_id), [])
                self.assertEqual(vector_db.deleted_workspaces, [(settings.child_collection, "demo")])
                self.assertEqual(list((settings.markdown_dir / "demo").glob("*")), [])

    def test_eval_main_writes_outputs_with_fake_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(os.environ, {}, clear=True):
                root = Path(tmp_dir)
                settings = load_settings(root_dir=root, use_env=False)
                dataset_path = root / "eval_dataset.jsonl"
                output_json = root / "results" / "results.json"
                output_report = root / "results" / "report.md"
                dataset_path.write_text(
                    json.dumps(
                        {
                            "case_id": "case-1",
                            "suite": "smoke",
                            "workspace_id": "demo",
                            "user_query": "What does the paper say?",
                            "intent_label": "general_retrieval",
                            "expected_paper_ids": ["paper-1"],
                            "expected_sections": ["Introduction"],
                            "forbidden_paper_ids": [],
                            "fallback_expected": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                args = SimpleNamespace(
                    dataset=str(dataset_path),
                    output_json=str(output_json),
                    output_report=str(output_report),
                    template=str(root / "missing_template.md.j2"),
                    limit=0,
                    skip_judge=True,
                    skip_calibration=True,
                    baseline="full",
                    ablation_baseline="",
                    min_pass_rate=0.7,
                    min_metric=[],
                    no_fail_on_threshold=False,
                )
                fake_result = {
                    "case_id": "case-1",
                    "suite": "smoke",
                    "status": "passed",
                    "passed": True,
                    "failure_reason": "",
                    "query": "What does the paper say?",
                    "intent_match": True,
                    "paper_coverage": True,
                    "section_hit": True,
                    "fallback_expected": False,
                    "fallback_detected": False,
                    "forbidden_hit": False,
                    "latency_ms": 10,
                    "predicted_intent": "general_retrieval",
                    "expected_intent": "general_retrieval",
                    "final_answer": "Supported answer from the fake runtime.",
                    "retrieved_chunks": ["chunk-1"],
                    "retrieved_parent_chunks": ["paper-1_parent_0"],
                    "retrieved_paper_ids": ["paper-1"],
                    "retrieved_sections": ["Introduction"],
                    "paper_profiles": [{"paper_id": "paper-1", "title": "Paper One"}],
                    "hard_metrics": {
                        "intent_routing": True,
                        "paper_coverage": True,
                        "document_isolation": None,
                        "valid_fallback": None,
                        "evidence_hit_at_3": None,
                        "evidence_hit_at_5": None,
                        "section_targeting": True,
                    },
                    "judge_results": {"faithfulness": None, "cross_doc_alignment": None, "valid_fallback": None},
                    "turn_result": {"rerank_backend": "fake", "verification_status": "pass"},
                }

                with mock.patch.object(eval_module, "parse_args", return_value=args):
                    with mock.patch.object(eval_module, "get_settings", return_value=settings):
                        with mock.patch.object(eval_module, "run_dataset_cases", return_value=[fake_result]):
                            exit_code = eval_module.main()

                payload = json.loads(output_json.read_text(encoding="utf-8"))
                self.assertEqual(payload["metadata"]["baseline"], "full")
                self.assertTrue(payload["metadata"]["gate"]["passed"])
                self.assertEqual(len(payload["results"]), 1)
                self.assertTrue(payload["results"][0]["passed"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Evaluation Report", output_report.read_text(encoding="utf-8"))

    def test_eval_main_no_fail_on_threshold_returns_success_for_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(os.environ, {}, clear=True):
                root = Path(tmp_dir)
                settings = load_settings(root_dir=root, use_env=False)
                dataset_path = root / "eval_dataset.jsonl"
                output_json = root / "results" / "results.json"
                output_report = root / "results" / "report.md"
                dataset_path.write_text(
                    json.dumps(
                        {
                            "case_id": "case-1",
                            "suite": "smoke",
                            "workspace_id": "demo",
                            "user_query": "What does the paper say?",
                            "intent_label": "general_retrieval",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                args = SimpleNamespace(
                    dataset=str(dataset_path),
                    output_json=str(output_json),
                    output_report=str(output_report),
                    template=str(root / "missing_template.md.j2"),
                    limit=0,
                    skip_judge=True,
                    skip_calibration=True,
                    baseline="full",
                    ablation_baseline="",
                    min_pass_rate=0.7,
                    min_metric=[],
                    no_fail_on_threshold=True,
                )
                fake_result = {
                    "case_id": "case-1",
                    "suite": "smoke",
                    "status": "failed",
                    "passed": False,
                    "failure_reason": "intent_routing_failed",
                    "query": "What does the paper say?",
                    "intent_match": False,
                    "latency_ms": 10,
                    "predicted_intent": "single_doc_close_reading",
                    "expected_intent": "general_retrieval",
                    "final_answer": "Wrong answer.",
                    "hard_metrics": {
                        "intent_routing": False,
                        "document_isolation": None,
                        "valid_fallback": None,
                        "evidence_hit_at_5": None,
                        "section_targeting": None,
                    },
                    "judge_results": {"faithfulness": None, "cross_doc_alignment": None, "valid_fallback": None},
                    "turn_result": {"rerank_backend": "fake", "verification_status": "unknown"},
                }

                with mock.patch.object(eval_module, "parse_args", return_value=args):
                    with mock.patch.object(eval_module, "get_settings", return_value=settings):
                        with mock.patch.object(eval_module, "run_dataset_cases", return_value=[fake_result]):
                            exit_code = eval_module.main()

                payload = json.loads(output_json.read_text(encoding="utf-8"))
                self.assertFalse(payload["metadata"]["gate"]["passed"])
                self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
