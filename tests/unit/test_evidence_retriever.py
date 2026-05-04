import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from agentic_rag.services.evidence_retriever import EvidenceRetriever
from agentic_rag.settings import load_settings
from agentic_rag.storage.parent_store import ParentStoreManager


class _FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, query, k, score_threshold, filter):
        return self.documents[:k]


class _FakeVectorDb:
    def get_filter(self, workspace_id, paper_id=None, section=None):
        return {"workspace_id": workspace_id, "paper_id": paper_id, "section": section}


class _FakeReranker:
    is_available = True

    def rerank(self, query, documents, top_n=None):
        return [type("Result", (), {"index": 1, "relevance_score": 0.9})(), type("Result", (), {"index": 0, "relevance_score": 0.4})()]


class _FailingReranker:
    is_available = True

    def rerank(self, query, documents, top_n=None):
        raise RuntimeError("boom")


class EvidenceRetrieverTests(unittest.TestCase):
    def test_search_uses_siliconflow_scores_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = load_settings(root_dir=Path(tmp_dir), use_env=False)
            parent_store = ParentStoreManager(settings)
            parent_store.save("demo", "paper-1", "paper-1_parent_0", "alpha content", {"paper_id": "paper-1", "section": "Intro", "source": "paper.pdf", "page_start": 1, "page_end": 1, "content_type": "paragraph"})
            parent_store.save("demo", "paper-1", "paper-1_parent_1", "beta content", {"paper_id": "paper-1", "section": "Method", "source": "paper.pdf", "page_start": 2, "page_end": 3, "content_type": "algorithm"})
            documents = [
                Document(page_content="alpha child", metadata={"parent_id": "paper-1_parent_0", "paper_id": "paper-1", "child_id": "paper-1_parent_0_child_0"}),
                Document(page_content="beta child", metadata={"parent_id": "paper-1_parent_1", "paper_id": "paper-1", "child_id": "paper-1_parent_1_child_0"}),
            ]
            retriever = EvidenceRetriever(settings, _FakeCollection(documents), _FakeVectorDb(), parent_store, _FakeReranker())

            result = retriever.search("method", workspace_id="demo", paper_id="paper-1", limit=2)

            self.assertEqual(result["rerank_backend"], "siliconflow")
            self.assertEqual(result["records"][0]["parent_id"], "paper-1_parent_1")

    def test_search_falls_back_to_heuristic_when_reranker_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = load_settings(root_dir=Path(tmp_dir), use_env=False)
            parent_store = ParentStoreManager(settings)
            parent_store.save("demo", "paper-1", "paper-1_parent_0", "blockchain workflow", {"paper_id": "paper-1", "section": "System Procedure", "section_norm": "system procedure", "source": "paper.pdf", "page_start": 1, "page_end": 2, "content_type": "algorithm"})
            document = Document(page_content="blockchain workflow", metadata={"parent_id": "paper-1_parent_0", "paper_id": "paper-1", "child_id": "paper-1_parent_0_child_0"})
            retriever = EvidenceRetriever(settings, _FakeCollection([document]), _FakeVectorDb(), parent_store, _FailingReranker())

            result = retriever.search("workflow", workspace_id="demo", paper_id="paper-1", limit=1, query_profile={"is_process": True})

            self.assertEqual(result["rerank_backend"], "heuristic_fallback")
            self.assertEqual(result["records"][0]["parent_id"], "paper-1_parent_0")

    def test_heuristic_rerank_uses_content_type_hints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = load_settings(root_dir=Path(tmp_dir), use_env=False)
            parent_store = ParentStoreManager(settings)
            parent_store.save(
                "demo",
                "paper-1",
                "paper-1_parent_0",
                "The experiment reports runtime and accuracy.",
                {
                    "paper_id": "paper-1",
                    "section": "Evaluation",
                    "section_norm": "evaluation",
                    "source": "paper.pdf",
                    "page_start": 7,
                    "page_end": 8,
                    "content_type": "paragraph",
                },
            )
            parent_store.save(
                "demo",
                "paper-1",
                "paper-1_parent_1",
                "Table 2. Runtime comparison across schemes.",
                {
                    "paper_id": "paper-1",
                    "section": "Evaluation",
                    "section_norm": "evaluation",
                    "source": "paper.pdf",
                    "page_start": 7,
                    "page_end": 8,
                    "content_type": "table",
                    "table_caption": "Table 2. Runtime comparison across schemes.",
                },
            )
            documents = [
                Document(page_content="experiment runtime", metadata={"parent_id": "paper-1_parent_0", "paper_id": "paper-1", "child_id": "paper-1_parent_0_child_0"}),
                Document(page_content="runtime comparison", metadata={"parent_id": "paper-1_parent_1", "paper_id": "paper-1", "child_id": "paper-1_parent_1_child_0"}),
            ]
            retriever = EvidenceRetriever(settings, _FakeCollection(documents), _FakeVectorDb(), parent_store, _FailingReranker())

            result = retriever.search(
                "Compare runtime performance",
                workspace_id="demo",
                paper_id="paper-1",
                limit=2,
                query_profile={"is_performance": True, "content_type_hints": ["table", "caption"]},
            )

            self.assertEqual(result["records"][0]["parent_id"], "paper-1_parent_1")
            self.assertEqual(result["records"][0]["content_type"], "table")

    def test_algorithmic_search_prioritizes_formula_refined_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = load_settings(root_dir=Path(tmp_dir), use_env=False)
            parent_store = ParentStoreManager(settings)
            parent_store.save(
                "demo",
                "paper-1",
                "paper-1_parent_0",
                "general construction overview",
                {
                    "paper_id": "paper-1",
                    "section": "Main Idea",
                    "section_norm": "main idea",
                    "source": "paper.pdf",
                    "page_start": 3,
                    "page_end": 3,
                    "content_type": "paragraph",
                },
            )
            parent_store.save(
                "demo",
                "paper-1",
                "paper-1_formula_refined_pages_7_7",
                "Definition 6 consists of nine algorithms: Setup, EKGen, DKGen, Enc, Tag, Match, ZKProof, ZKVer, Dec.",
                {
                    "paper_id": "paper-1",
                    "section": "Formula refinement pages 7-7",
                    "section_norm": "formula refinement pages 7 7",
                    "source": "paper.pdf",
                    "page_start": 7,
                    "page_end": 7,
                    "content_type": "formula_refined",
                    "equation_dense": True,
                },
            )
            documents = [
                Document(page_content="construction overview", metadata={"parent_id": "paper-1_parent_0", "paper_id": "paper-1", "child_id": "paper-1_parent_0_child_0"}),
                Document(page_content="Setup EKGen DKGen Enc Tag Match ZKProof ZKVer Dec", metadata={"parent_id": "paper-1_formula_refined_pages_7_7", "paper_id": "paper-1", "child_id": "paper-1_formula_refined_pages_7_7_child_0"}),
            ]
            retriever = EvidenceRetriever(settings, _FakeCollection(documents), _FakeVectorDb(), parent_store, _FailingReranker())

            result = retriever.search(
                "完整算法 Setup EKGen DKGen Enc Tag Match ZKProof ZKVer Dec",
                workspace_id="demo",
                paper_id="paper-1",
                limit=2,
                query_profile={"is_algorithmic": True, "content_type_hints": ["algorithm", "formula"]},
            )

            self.assertEqual(result["records"][0]["parent_id"], "paper-1_formula_refined_pages_7_7")
            self.assertEqual(result["records"][0]["content_type"], "formula_refined")


if __name__ == "__main__":
    unittest.main()
