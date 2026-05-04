from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from agentic_rag.settings import Settings
from agentic_rag.utils.ids import resolve_paper_id
from agentic_rag.utils.pdf import pdf_to_markdown
from agentic_rag.utils.text import clean_markdown_text


@dataclass
class ParsedSection:
    heading: str
    content: str
    section_type: str
    page_start: int | None = None
    page_end: int | None = None
    block_type: str = "paragraph"


@dataclass
class ParsedDocument:
    paper_id: str
    source_name: str
    markdown_path: Path
    title: str
    sections: List[ParsedSection] = field(default_factory=list)


class PdfParserAdapter(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def parse(self, pdf_path: str | Path, output_dir: str | Path) -> ParsedDocument:
        raise NotImplementedError

    def _extract_sections(self, markdown_text: str) -> List[ParsedSection]:
        lines = markdown_text.splitlines()
        sections: List[ParsedSection] = []
        current_heading = "Document Overview"
        current_type = "front_matter"
        current_page = 1
        section_start_page = 1
        buffer: List[str] = []

        def flush():
            content = "\n".join(buffer).strip()
            if content:
                sections.append(
                    ParsedSection(
                        heading=current_heading,
                        content=content,
                        section_type=current_type,
                        page_start=section_start_page,
                        page_end=current_page,
                    )
                )

        for line in lines:
            stripped = line.strip()
            page_match = re.match(r"(?i)^#*\s*(?:page\s+)?(\d{1,4})\s*$", stripped)
            if page_match:
                current_page = int(page_match.group(1))
                if not any(part.strip() for part in buffer):
                    section_start_page = current_page
                continue

            if line.startswith("#"):
                flush()
                buffer.clear()
                current_heading = line.lstrip("#").strip() or current_heading
                current_type = self._classify_heading(current_heading)
                section_start_page = current_page
                continue

            buffer.append(line)

        flush()
        return sections

    def _assign_pdf_page_ranges(self, sections: List[ParsedSection], pdf_path: Path) -> List[ParsedSection]:
        if not sections or pdf_path.suffix.lower() != ".pdf":
            return sections

        try:
            import pymupdf
        except ImportError:
            return sections

        try:
            doc = pymupdf.open(pdf_path)
            page_texts = [page.get_text("text") for page in doc]
            doc.close()
        except Exception:
            return sections

        if not page_texts:
            return sections

        page_norms = [self._normalize_match_text(text) for text in page_texts]
        page_tokens = [set(text.split()) for text in page_norms]
        start_pages: list[int] = []
        search_from = 0

        for section in sections:
            heading_norm = self._normalize_match_text(section.heading)
            content_norm = self._normalize_match_text(section.content[:1200])
            content_tokens = [token for token in content_norm.split() if len(token) > 2][:80]
            best_index = search_from
            best_score = -1

            for page_index in range(search_from, len(page_norms)):
                page_norm = page_norms[page_index]
                score = 0
                if heading_norm and heading_norm in page_norm:
                    score += 1000
                if content_tokens:
                    score += len(set(content_tokens) & page_tokens[page_index])
                if score > best_score:
                    best_score = score
                    best_index = page_index

            start_page = best_index + 1
            start_pages.append(start_page)
            search_from = max(search_from, best_index)

        page_count = len(page_texts)
        for index, section in enumerate(sections):
            start = start_pages[index]
            next_start = start_pages[index + 1] if index + 1 < len(start_pages) else page_count
            end = max(start, min(page_count, next_start))
            section.page_start = start
            section.page_end = end
        return sections

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _classify_heading(heading: str) -> str:
        normalized = heading.strip().lower()
        if "abstract" in normalized:
            return "abstract"
        if "introduction" in normalized:
            return "introduction"
        if any(key in normalized for key in ("method", "approach", "model", "framework", "procedure", "workflow")):
            return "method"
        if any(key in normalized for key in ("experiment", "evaluation", "result")):
            return "experiment"
        if any(key in normalized for key in ("conclusion", "discussion")):
            return "conclusion"
        if "appendix" in normalized:
            return "appendix"
        return "section"


class PymuPdf4LlmParserAdapter(PdfParserAdapter):
    def parse(self, pdf_path: str | Path, output_dir: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paper_id = resolve_paper_id(pdf_path.stem)

        if pdf_path.suffix.lower() == ".md":
            markdown_path = output_dir / f"{paper_id}.md"
            if pdf_path.resolve() != markdown_path.resolve():
                markdown_path.write_text(pdf_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            markdown_path = pdf_to_markdown(pdf_path, output_dir, output_stem=paper_id)
        text = markdown_path.read_text(encoding="utf-8")
        sections = self._extract_sections(text)
        sections = self._assign_pdf_page_ranges(sections, pdf_path)
        title = sections[0].heading if sections else pdf_path.stem
        return ParsedDocument(
            paper_id=paper_id,
            source_name=pdf_path.name,
            markdown_path=markdown_path,
            title=title,
            sections=sections,
        )


class DoclingParserAdapter(PdfParserAdapter):
    def parse(self, pdf_path: str | Path, output_dir: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paper_id = resolve_paper_id(pdf_path.stem)

        if pdf_path.suffix.lower() == ".md":
            markdown_path = output_dir / f"{paper_id}.md"
            if pdf_path.resolve() != markdown_path.resolve():
                markdown_path.write_text(pdf_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as exc:
                raise ImportError(
                    "Docling is not installed. Run `pip install docling` or reinstall from pyproject extras."
                ) from exc

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.settings.docling_do_ocr
            if self.settings.docling_do_formula_enrichment:
                if not hasattr(pipeline_options, "do_formula_enrichment"):
                    raise RuntimeError(
                        "The installed Docling version does not support formula enrichment. "
                        "Upgrade Docling or set DOCLING_DO_FORMULA_ENRICHMENT=false."
                    )
                pipeline_options.do_formula_enrichment = True
            if self.settings.docling_artifacts_path:
                pipeline_options.artifacts_path = str(self.settings.docling_artifacts_path)

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            try:
                result = converter.convert(str(pdf_path))
            except Exception as exc:
                message = str(exc)
                if "locate the files on the Hub" in message:
                    artifacts_path = self.settings.docling_artifacts_path or ".docling_artifacts"
                    raise RuntimeError(
                        "Docling models are not cached locally yet. Pre-download them first with:\n"
                        f"  .\\.venv\\Scripts\\python -m docling.cli.models download -o {artifacts_path}\n"
                        "Then set DOCLING_ARTIFACTS_PATH in .env to that same folder and restart the app."
                    ) from exc
                raise
            markdown_text = clean_markdown_text(result.document.export_to_markdown())
            markdown_path = output_dir / f"{paper_id}.md"
            markdown_path.write_text(markdown_text, encoding="utf-8")

        text = markdown_path.read_text(encoding="utf-8")
        sections = self._extract_sections(text)
        sections = self._assign_pdf_page_ranges(sections, pdf_path)
        title = sections[0].heading if sections else pdf_path.stem
        return ParsedDocument(
            paper_id=paper_id,
            source_name=pdf_path.name,
            markdown_path=markdown_path,
            title=title,
            sections=sections,
        )


class MinerUParserAdapter(PdfParserAdapter):
    def parse(self, pdf_path: str | Path, output_dir: str | Path) -> ParsedDocument:
        raise NotImplementedError("MinerU parser is not enabled in the lightweight V1 path.")


def create_pdf_parser(settings: Settings) -> PdfParserAdapter:
    backend = settings.pdf_parser_backend
    if backend == "pymupdf4llm":
        return PymuPdf4LlmParserAdapter(settings)
    if backend == "docling":
        return DoclingParserAdapter(settings)
    if backend == "mineru":
        return MinerUParserAdapter(settings)
    raise ValueError(f"Unsupported PDF_PARSER_BACKEND: {backend}")
