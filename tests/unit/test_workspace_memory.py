import tempfile
import unittest
from pathlib import Path

from agentic_rag.settings import load_settings
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore


class WorkspaceMemoryTests(unittest.TestCase):
    def test_register_and_list_papers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-1", "title": "Demo Paper"})
            papers = store.list_papers("demo")
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0]["paper_id"], "paper-1")

    def test_search_helpers_return_relevant_memory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.register_paper("demo", {"paper_id": "paper-1", "title": "Demo Paper"})
            store.save_paper_profile("demo", "paper-1", {"paper_id": "paper-1", "core_method": "blockchain access control"})
            store.save_working_memory_snapshot(
                "demo",
                {"focus_paper_id": "paper-1", "current_research_question": "How does blockchain help access control?"},
            )
            store.record_interaction("demo", {"query": "Explain the blockchain workflow", "intent_type": "general_retrieval"})

            profile_hits = store.search_paper_profiles("demo", "blockchain")
            interaction_hits = store.search_interactions("demo", "workflow")
            working_memory = store.search_working_memory("demo", "access control")

            self.assertEqual(profile_hits[0]["paper_id"], "paper-1")
            self.assertEqual(interaction_hits[0]["intent_type"], "general_retrieval")
            self.assertEqual(working_memory["focus_paper_id"], "paper-1")

    def test_semantic_and_episodic_layers_are_searchable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.save_semantic_fact(
                "demo",
                "paper_aliases::paper-1",
                {
                    "kind": "paper_aliases",
                    "paper_id": "paper-1",
                    "aliases": ["be-pbac"],
                    "title": "BE-PBAC",
                },
            )
            store.record_interaction(
                "demo",
                {
                    "query": "Compare BE-PBAC and PM-ABE",
                    "intent_type": "cross_doc_comparison",
                    "paper_ids": ["paper-1", "paper-2"],
                    "retrieved_sections": ["Performance Evaluation"],
                    "verification_status": "pass",
                    "artifact_kind": "build_comparison_table",
                },
            )

            semantic_hits = store.search_semantic_memory("demo", "BE-PBAC")
            episodic_hits = store.search_episodic_memory("demo", "comparison")

            self.assertEqual(semantic_hits[0]["paper_id"], "paper-1")
            self.assertEqual(episodic_hits[0]["artifact_kind"], "build_comparison_table")
            self.assertIn("created_at", episodic_hits[0])

    def test_save_paper_profile_also_writes_semantic_fact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = WorkspaceMemoryStore(settings)
            store.save_paper_profile(
                "demo",
                "paper-1",
                {
                    "paper_id": "paper-1",
                    "title": "Demo Paper",
                    "core_method": "bilateral access control",
                    "source_sections": ["Construction"],
                },
            )

            semantic_hits = store.search_semantic_memory("demo", "bilateral")

            self.assertEqual(semantic_hits[0]["kind"], "paper_profile")
            self.assertEqual(semantic_hits[0]["paper_id"], "paper-1")


if __name__ == "__main__":
    unittest.main()
