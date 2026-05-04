import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentic_rag.parsers import create_pdf_parser
from agentic_rag.services.formula_refinement_service import FormulaRefinementService
from agentic_rag.settings import load_settings
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore


class FormulaRefinementServiceTests(unittest.TestCase):
    def test_lists_formula_candidates_from_markdown_and_source_pdf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            workspace_memory = WorkspaceMemoryStore(settings)
            workspace = "demo"
            paper_id = "paper-1"
            md_path = settings.markdown_dir / workspace / f"{paper_id}.md"
            pdf_path = settings.source_docs_dir / workspace / f"{paper_id}.pdf"
            md_path.parent.mkdir(parents=True)
            pdf_path.parent.mkdir(parents=True)
            md_path.write_text(
                "1\n## Intro\nText\n2\n## Math\n<!-- formula-not-decoded -->\n",
                encoding="utf-8",
            )
            pdf_path.write_bytes(b"%PDF-placeholder")
            workspace_memory.register_paper(
                workspace,
                {
                    "paper_id": paper_id,
                    "title": "Paper",
                    "source_name": "paper.pdf",
                    "source_path": str(pdf_path),
                    "markdown_path": str(md_path),
                    "sections": ["Intro", "Math"],
                },
            )
            rag_system = SimpleNamespace(
                workspace_memory=workspace_memory,
                parser=create_pdf_parser(settings),
            )
            service = FormulaRefinementService(rag_system, settings)

            candidates = service.list_candidates(workspace)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].paper_id, paper_id)
            self.assertEqual(candidates[0].formula_marker_count, 1)
            self.assertEqual(candidates[0].pages, [2])

    def test_refine_workspace_uses_max_pages_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            service = FormulaRefinementService(SimpleNamespace(workspace_memory=None, parser=None), settings)
            candidate = SimpleNamespace(
                paper_id="paper-1",
                title="Paper",
                source_path=Path("paper.pdf"),
                markdown_path=Path("paper.md"),
                formula_marker_count=2,
                pages=[1, 2, 4],
            )
            service._convert_page_range = mock.Mock(return_value="refined formula")
            service.rag_system = SimpleNamespace(
                parent_store=SimpleNamespace(save=lambda *args, **kwargs: None),
                vector_db=SimpleNamespace(get_collection=lambda name: SimpleNamespace(add_documents=lambda docs: None)),
                collection_name=settings.child_collection,
            )
            progress_updates = []

            report = service.refine_candidate(
                "demo",
                candidate,
                max_pages=2,
                progress_callback=lambda value, desc: progress_updates.append((value, desc)),
            )

            self.assertEqual(report["pages_requested"], [1, 2])
            self.assertEqual(report["pages_refined"], [1, 2])
            self.assertEqual(report["pages_skipped_existing"], [])
            self.assertEqual(report["ranges_refined"], [(1, 1), (2, 2)])
            self.assertEqual(report["merged_pages"], [])
            self.assertTrue(any(value > 0.05 for value, _desc in progress_updates))

    def test_refine_candidate_unlimited_skips_existing_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            service = FormulaRefinementService(SimpleNamespace(workspace_memory=None, parser=None), settings)
            candidate = SimpleNamespace(
                paper_id="paper-1",
                title="Paper",
                source_path=Path("paper.pdf"),
                markdown_path=Path("paper.md"),
                formula_marker_count=4,
                pages=[5, 8, 9, 10],
            )
            paper_dir = settings.formula_refinement_path / "demo" / "paper-1"
            paper_dir.mkdir(parents=True)
            (paper_dir / "pages_5_5.md").write_text("already refined", encoding="utf-8")
            (paper_dir / "pages_8_9.md").write_text("already refined", encoding="utf-8")
            service._convert_page_range = mock.Mock(return_value="refined formula")
            service.rag_system = SimpleNamespace(
                parent_store=SimpleNamespace(save=lambda *args, **kwargs: None),
                vector_db=SimpleNamespace(get_collection=lambda name: SimpleNamespace(add_documents=lambda docs: None)),
                collection_name=settings.child_collection,
            )

            report = service.refine_candidate("demo", candidate, max_pages=0)

            self.assertEqual(report["pages_requested"], [5, 8, 9, 10])
            self.assertEqual(report["pages_skipped_existing"], [5, 8, 9])
            self.assertEqual(report["pages_refined"], [10])
            self.assertEqual(report["ranges_refined"], [(10, 10)])
            service._convert_page_range.assert_called_once_with(Path("paper.pdf"), 10, 10)

    def test_merge_refinements_updates_markdown_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            workspace_memory = WorkspaceMemoryStore(settings)
            workspace = "demo"
            paper_id = "paper-1"
            md_path = settings.markdown_dir / workspace / f"{paper_id}.md"
            md_path.parent.mkdir(parents=True)
            md_path.write_text(
                "1\n## Intro\nText\n2\n## Math\n<!-- formula-not-decoded -->\n",
                encoding="utf-8",
            )
            refined_dir = settings.formula_refinement_path / workspace / paper_id
            refined_dir.mkdir(parents=True)
            (refined_dir / "pages_2_2.md").write_text("## Math\n\n$$x = y$$", encoding="utf-8")
            workspace_memory.register_paper(
                workspace,
                {
                    "paper_id": paper_id,
                    "title": "Paper",
                    "source_name": "paper.pdf",
                    "source_path": "",
                    "markdown_path": str(md_path),
                    "sections": ["Intro", "Math"],
                },
            )
            rag_system = SimpleNamespace(
                workspace_memory=workspace_memory,
                parser=create_pdf_parser(settings),
            )
            service = FormulaRefinementService(rag_system, settings)

            first_report = service.merge_refinements(workspace, paper_id=paper_id)
            second_report = service.merge_refinements(workspace, paper_id=paper_id)
            merged_text = md_path.read_text(encoding="utf-8")

            self.assertEqual(first_report["merged_count"], 1)
            self.assertEqual(second_report["merged_count"], 1)
            self.assertEqual(merged_text.count("Formula Refinement Merge"), 1)
            self.assertEqual(merged_text.count("formula-not-decoded"), 0)
            self.assertIn("formula-refined-see-merged-section", merged_text)
            self.assertIn("$$x = y$$", merged_text)

    def test_refine_candidate_merges_generated_refinement_into_markdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            md_path = settings.markdown_dir / "demo" / "paper-1.md"
            md_path.parent.mkdir(parents=True)
            md_path.write_text("1\n## Math\n<!-- formula-not-decoded -->\n", encoding="utf-8")
            service = FormulaRefinementService(SimpleNamespace(workspace_memory=None, parser=None), settings)
            candidate = SimpleNamespace(
                paper_id="paper-1",
                title="Paper",
                source_path=Path("paper.pdf"),
                markdown_path=md_path,
                formula_marker_count=1,
                pages=[1],
            )
            service._convert_page_range = mock.Mock(return_value="$$a = b$$")
            service.rag_system = SimpleNamespace(
                parent_store=SimpleNamespace(save=lambda *args, **kwargs: None),
                vector_db=SimpleNamespace(get_collection=lambda name: SimpleNamespace(add_documents=lambda docs: None)),
                collection_name=settings.child_collection,
            )

            report = service.refine_candidate("demo", candidate, max_pages=0)
            merged_text = md_path.read_text(encoding="utf-8")

            self.assertEqual(report["merged_pages"], [1])
            self.assertEqual(report["formula_markers_after_merge"], 0)
            self.assertIn("Formula Refinement Merge", merged_text)
            self.assertIn("$$a = b$$", merged_text)
            self.assertNotIn("formula-not-decoded", merged_text)

    def test_refine_candidate_splits_large_formula_parent_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            object.__setattr__(settings, "max_parent_size", 220)
            object.__setattr__(settings, "child_chunk_size", 120)
            object.__setattr__(settings, "child_chunk_overlap", 20)
            md_path = settings.markdown_dir / "demo" / "paper-1.md"
            md_path.parent.mkdir(parents=True)
            md_path.write_text("1\n## Math\n<!-- formula-not-decoded -->\n", encoding="utf-8")
            service = FormulaRefinementService(SimpleNamespace(workspace_memory=None, parser=None), settings)
            candidate = SimpleNamespace(
                paper_id="paper-1",
                title="Paper",
                source_path=Path("paper.pdf"),
                markdown_path=md_path,
                formula_marker_count=1,
                pages=[1],
            )
            formula_lines = "\n".join(f"x_{{{index}}} = y_{{{index}}} + z_{{{index}}}" for index in range(30))
            service._convert_page_range = mock.Mock(return_value=f"$$\n{formula_lines}\n$$")
            saved_parents = []
            indexed_children = []
            service.rag_system = SimpleNamespace(
                parent_store=SimpleNamespace(save=lambda *args, **kwargs: saved_parents.append(args)),
                vector_db=SimpleNamespace(get_collection=lambda name: SimpleNamespace(add_documents=lambda docs: indexed_children.extend(docs))),
                collection_name=settings.child_collection,
            )

            report = service.refine_candidate("demo", candidate, max_pages=0)

            self.assertGreater(len(report["added_parent_ids"]), 1)
            self.assertTrue(any(parent_id.endswith("_part_0") for parent_id in report["added_parent_ids"]))
            self.assertTrue(all(saved[3].count("$$") % 2 == 0 for saved in saved_parents))
            self.assertTrue(all(doc.page_content.count("$$") % 2 == 0 for doc in indexed_children))


if __name__ == "__main__":
    unittest.main()
