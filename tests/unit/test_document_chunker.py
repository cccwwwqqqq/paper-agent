import tempfile
import unittest
from pathlib import Path

from agentic_rag.document_chunker import DocumentChuncker
from agentic_rag.parsers import ParsedDocument, ParsedSection
from agentic_rag.settings import load_settings


class DocumentChunkerTests(unittest.TestCase):
    def test_child_chunks_preserve_formula_and_caption_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            chunker = DocumentChuncker(settings)
            markdown_path = root / "demo.md"
            markdown_path.write_text("# Demo", encoding="utf-8")
            parsed_document = ParsedDocument(
                paper_id="paper-1",
                source_name="paper.pdf",
                markdown_path=markdown_path,
                title="Demo",
                sections=[
                    ParsedSection(
                        heading="Detailed Construction",
                        content="Setup(x)\nKeyGen(y)\nformula-not-decoded\nlambda = e(g, h)\na_{i} = b_{i}\n",
                        section_type="method",
                        page_start=2,
                        page_end=3,
                    ),
                    ParsedSection(
                        heading="Figure 2",
                        content="Figure 2. System workflow for online authorization.\nThe protocol has three stages.",
                        section_type="section",
                        page_start=4,
                        page_end=4,
                    ),
                    ParsedSection(
                        heading="Table 1",
                        content="Table 1. Runtime comparison across schemes.\nPM-ABE | 10ms | 12ms",
                        section_type="experiment",
                        page_start=5,
                        page_end=5,
                    ),
                ],
            )

            parent_chunks, child_chunks = chunker.create_chunks_single(parsed_document, "demo")

            self.assertEqual(len(parent_chunks), 3)
            formula_child = next(doc for doc in child_chunks if doc.metadata.get("section") == "Detailed Construction")
            figure_child = next(doc for doc in child_chunks if doc.metadata.get("section") == "Figure 2")
            table_child = next(doc for doc in child_chunks if doc.metadata.get("section") == "Table 1")

            self.assertEqual(formula_child.metadata["content_type"], "formula")
            self.assertTrue(formula_child.metadata["equation_dense"])
            self.assertEqual(figure_child.metadata["content_type"], "figure")
            self.assertIn("Figure 2", figure_child.metadata["figure_caption"])
            self.assertEqual(table_child.metadata["content_type"], "table")
            self.assertIn("Table 1", table_child.metadata["table_caption"])

    def test_formula_aware_splitter_preserves_math_delimiters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            object.__setattr__(settings, "max_parent_size", 220)
            object.__setattr__(settings, "child_chunk_size", 120)
            object.__setattr__(settings, "child_chunk_overlap", 20)
            chunker = DocumentChuncker(settings)
            markdown_path = root / "demo.md"
            markdown_path.write_text("# Demo", encoding="utf-8")
            formula_lines = "\n".join(f"x_{{{index}}} = y_{{{index}}} + z_{{{index}}}" for index in range(30))
            parsed_document = ParsedDocument(
                paper_id="paper-1",
                source_name="paper.pdf",
                markdown_path=markdown_path,
                title="Demo",
                sections=[
                    ParsedSection(
                        heading="Formula Section",
                        content=f"Context before the equation.\n\n$$\n{formula_lines}\n$$\n\nContext after the equation.",
                        section_type="method",
                        page_start=1,
                        page_end=1,
                    ),
                ],
            )

            parent_chunks, child_chunks = chunker.create_chunks_single(parsed_document, "demo")

            self.assertGreater(len(parent_chunks), 1)
            self.assertTrue(all(doc.page_content.count("$$") % 2 == 0 for _parent_id, doc in parent_chunks))
            self.assertTrue(all(doc.page_content.count("$$") % 2 == 0 for doc in child_chunks))


if __name__ == "__main__":
    unittest.main()
