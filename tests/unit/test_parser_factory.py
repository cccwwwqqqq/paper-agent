import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentic_rag.parsers import create_pdf_parser
from agentic_rag.parsers.adapters import DoclingParserAdapter, ParsedSection, PymuPdf4LlmParserAdapter
from agentic_rag.settings import load_settings


class ParserFactoryTests(unittest.TestCase):
    def test_create_pymupdf_parser(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_settings(use_env=False)
            parser = create_pdf_parser(settings)
        self.assertIsInstance(parser, PymuPdf4LlmParserAdapter)

    def test_create_docling_parser(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_settings(use_env=False)
            object.__setattr__(settings, "pdf_parser_backend", "docling")
            parser = create_pdf_parser(settings)
        self.assertIsInstance(parser, DoclingParserAdapter)

    def test_extract_sections_tracks_standalone_page_numbers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_settings(use_env=False)
            parser = PymuPdf4LlmParserAdapter(settings)

        sections = parser._extract_sections(
            "1\n"
            "## Intro\n"
            "first page text\n"
            "2\n"
            "second page text\n"
            "## Method\n"
            "method text\n"
        )

        self.assertEqual(sections[0].heading, "Intro")
        self.assertEqual(sections[0].page_start, 1)
        self.assertEqual(sections[0].page_end, 2)
        self.assertEqual(sections[1].heading, "Method")
        self.assertEqual(sections[1].page_start, 2)
        self.assertEqual(sections[1].page_end, 2)

    def test_assign_pdf_page_ranges_uses_source_pdf_pages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            import pymupdf

            root = Path(tmp_dir)
            pdf_path = root / "sample.pdf"
            doc = pymupdf.open()
            for text in [
                "Title\nIntro\nThis is the first page.",
                "Method\nThis page explains the construction.",
                "Results\nThis page reports measurements.",
            ]:
                page = doc.new_page()
                page.insert_text((72, 72), text)
            doc.save(pdf_path)
            doc.close()

            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_settings(root_dir=root, use_env=False)
                parser = PymuPdf4LlmParserAdapter(settings)

            sections = [
                ParsedSection("Intro", "This is the first page.", "section"),
                ParsedSection("Method", "This page explains the construction.", "section"),
                ParsedSection("Results", "This page reports measurements.", "section"),
            ]

            parser._assign_pdf_page_ranges(sections, pdf_path)

            self.assertEqual([(section.page_start, section.page_end) for section in sections], [(1, 2), (2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()
