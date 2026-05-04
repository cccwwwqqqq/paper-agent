from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from agentic_rag.settings import Settings
from agentic_rag.utils.markdown_chunking import split_formula_aware_documents
from agentic_rag.utils.text import clean_markdown_text

FORMULA_MARKER = "formula-not-decoded"
FORMULA_REFINED_POINTER = "<!-- formula-refined-see-merged-section -->"
MERGE_BLOCK_START = "<!-- formula-refinement-merge:start -->"
MERGE_BLOCK_END = "<!-- formula-refinement-merge:end -->"


@dataclass(frozen=True)
class FormulaRefinementCandidate:
    paper_id: str
    title: str
    source_path: Path
    markdown_path: Path
    formula_marker_count: int
    pages: list[int]


class FormulaRefinementService:
    def __init__(self, rag_system, settings: Settings):
        self.rag_system = rag_system
        self.settings = settings
        self.refinement_root = Path(settings.formula_refinement_path)
        self.refinement_root.mkdir(parents=True, exist_ok=True)

    def list_candidates(self, workspace_id: str, paper_id: str | None = None) -> list[FormulaRefinementCandidate]:
        candidates = []
        for paper in self.rag_system.workspace_memory.list_papers(workspace_id):
            if paper_id and paper.get("paper_id") != paper_id:
                continue

            markdown_path = Path(paper.get("markdown_path", ""))
            source_path = Path(paper.get("source_path", ""))
            if not markdown_path.exists() or not source_path.exists() or source_path.suffix.lower() != ".pdf":
                continue

            text = markdown_path.read_text(encoding="utf-8")
            marker_count = text.count(FORMULA_MARKER)
            if marker_count <= 0:
                continue

            pages = self._pages_with_formula_markers(text, source_path)
            candidates.append(
                FormulaRefinementCandidate(
                    paper_id=paper["paper_id"],
                    title=paper.get("title", paper["paper_id"]),
                    source_path=source_path,
                    markdown_path=markdown_path,
                    formula_marker_count=marker_count,
                    pages=pages,
                )
            )
        return candidates

    def refine_workspace(
        self,
        workspace_id: str,
        paper_id: str | None = None,
        max_pages: int | None = None,
        progress_callback=None,
    ) -> dict:
        max_pages = self._resolve_max_pages(max_pages)
        reports = []
        if progress_callback:
            progress_callback(0.01, "Finding formula-heavy pages")
        candidates = self.list_candidates(workspace_id, paper_id)
        if progress_callback:
            progress_callback(0.04, f"Found {len(candidates)} document(s) with formula placeholders")
        total_pages = sum(
            len(self._pages_to_refine(workspace_id, candidate, max_pages))
            for candidate in candidates
        )
        if progress_callback:
            progress_callback(
                0.05,
                f"{total_pages} formula page(s) queued for refinement",
            )
        completed_pages = 0
        for candidate in candidates:
            report = self.refine_candidate(
                workspace_id,
                candidate,
                max_pages=max_pages,
                progress_callback=progress_callback,
                completed_pages=completed_pages,
                total_pages=total_pages,
            )
            completed_pages += len(report.get("pages_refined", []))
            reports.append(report)
        if progress_callback:
            progress_callback(1.0, "Formula refinement complete")
        return {
            "workspace_id": workspace_id,
            "paper_id": paper_id or "",
            "candidate_count": len(reports),
            "reports": reports,
        }

    def refine_candidate(
        self,
        workspace_id: str,
        candidate: FormulaRefinementCandidate,
        *,
        max_pages: int | None = None,
        progress_callback=None,
        completed_pages: int = 0,
        total_pages: int = 0,
    ) -> dict:
        max_pages = self._resolve_max_pages(max_pages)
        pages = self._selected_pages(candidate.pages, max_pages)
        pages_to_refine = self._pages_to_refine(workspace_id, candidate, max_pages)
        skipped_pages = [page for page in pages if page not in pages_to_refine]
        ranges = [(page, page) for page in pages_to_refine]
        paper_dir = self.refinement_root / workspace_id / candidate.paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        added_parent_ids: list[str] = []
        output_paths: list[str] = []
        before_count = candidate.formula_marker_count
        refined_marker_count = 0

        for range_index, (start_page, end_page) in enumerate(ranges):
            if progress_callback:
                current = completed_pages + range_index
                total = max(total_pages, len(pages_to_refine), 1)
                progress_callback(
                    min(0.95, self._progress_for_page(current, total, 0.15)),
                    f"Starting formula refinement for page {start_page}",
                )
            refined_markdown = self._convert_page_range_with_progress(
                candidate.source_path,
                start_page,
                end_page,
                completed_pages + range_index,
                max(total_pages, len(pages_to_refine), 1),
                progress_callback,
            )
            refined_marker_count += refined_markdown.count(FORMULA_MARKER)
            output_path = paper_dir / f"pages_{start_page}_{end_page}.md"
            output_path.write_text(refined_markdown, encoding="utf-8")
            output_paths.append(str(output_path))

            parent_id = f"{candidate.paper_id}_formula_refined_pages_{start_page}_{end_page}"
            parent_doc = Document(
                page_content=f"# Formula refinement pages {start_page}-{end_page}\n\n{refined_markdown}".strip(),
                metadata={
                    "workspace_id": workspace_id,
                    "paper_id": candidate.paper_id,
                    "source": candidate.source_path.name,
                    "title": candidate.title,
                    "parent_id": parent_id,
                    "section": f"Formula refinement pages {start_page}-{end_page}",
                    "section_norm": f"formula refinement pages {start_page} {end_page}",
                    "section_type": "formula_refinement",
                    "block_type": "formula",
                    "content_type": "formula_refined",
                    "equation_dense": True,
                    "table_caption": "",
                    "figure_caption": "",
                    "page_start": start_page,
                    "page_end": end_page,
                },
            )
            parent_docs = split_formula_aware_documents(
                parent_doc,
                chunk_size=self.settings.max_parent_size,
                chunk_overlap=self.settings.child_chunk_overlap,
            )
            if not parent_docs:
                parent_docs = [parent_doc]

            collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
            for parent_idx, split_parent_doc in enumerate(parent_docs):
                split_parent_id = parent_id if len(parent_docs) == 1 else f"{parent_id}_part_{parent_idx}"
                split_parent_doc.metadata.update(parent_doc.metadata)
                split_parent_doc.metadata["parent_id"] = split_parent_id
                self.rag_system.parent_store.save(
                    workspace_id,
                    candidate.paper_id,
                    split_parent_id,
                    split_parent_doc.page_content,
                    split_parent_doc.metadata,
                )
                added_parent_ids.append(split_parent_id)

                child_docs = split_formula_aware_documents(
                    split_parent_doc,
                    chunk_size=self.settings.child_chunk_size,
                    chunk_overlap=self.settings.child_chunk_overlap,
                )
                for child_idx, child_doc in enumerate(child_docs):
                    child_doc.metadata.update(split_parent_doc.metadata)
                    child_doc.metadata["child_id"] = f"{split_parent_id}_child_{child_idx}"
                if child_docs:
                    collection.add_documents(child_docs)
            if progress_callback:
                done = completed_pages + range_index + 1
                total = max(total_pages, len(pages_to_refine), 1)
                progress_callback(
                    min(0.98, 0.05 + 0.9 * (done / total)),
                    f"Indexed refined formula page {start_page}",
                )

        report = {
            "paper_id": candidate.paper_id,
            "source_path": str(candidate.source_path),
            "pages_requested": pages,
            "pages_refined": pages_to_refine,
            "pages_skipped_existing": skipped_pages,
            "ranges_refined": ranges,
            "formula_markers_before": before_count,
            "formula_markers_in_refined_output": refined_marker_count,
            "added_parent_ids": added_parent_ids,
            "output_paths": output_paths,
        }
        merge_report = self._merge_candidate_refinements(workspace_id, candidate)
        report.update(
            {
                "merged_markdown_path": merge_report.get("markdown_path", ""),
                "merged_pages": merge_report.get("merged_pages", []),
                "formula_markers_after_merge": merge_report.get("formula_markers_after_merge", before_count),
            }
        )
        (paper_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def merge_refinements(self, workspace_id: str, paper_id: str | None = None) -> dict:
        reports = []
        workspace_memory = getattr(self.rag_system, "workspace_memory", None)
        if not workspace_memory:
            return {"workspace_id": workspace_id, "paper_id": paper_id or "", "merged_count": 0, "reports": reports}

        for paper in workspace_memory.list_papers(workspace_id):
            if paper_id and paper.get("paper_id") != paper_id:
                continue
            markdown_path = Path(paper.get("markdown_path", ""))
            if not markdown_path.exists():
                continue
            source_path = Path(paper.get("source_path", ""))
            markdown_text = markdown_path.read_text(encoding="utf-8")
            pages = self._pages_with_formula_markers(markdown_text, source_path if source_path.exists() else None)
            candidate = FormulaRefinementCandidate(
                paper_id=paper["paper_id"],
                title=paper.get("title", paper["paper_id"]),
                source_path=source_path,
                markdown_path=markdown_path,
                formula_marker_count=markdown_text.count(FORMULA_MARKER),
                pages=pages,
            )
            report = self._merge_candidate_refinements(workspace_id, candidate)
            if report.get("merged_pages"):
                reports.append(report)

        return {
            "workspace_id": workspace_id,
            "paper_id": paper_id or "",
            "merged_count": len(reports),
            "reports": reports,
        }

    def _merge_candidate_refinements(self, workspace_id: str, candidate: FormulaRefinementCandidate) -> dict:
        markdown_path = Path(candidate.markdown_path)
        paper_dir = self.refinement_root / workspace_id / candidate.paper_id
        refined_files = self._refined_page_files(paper_dir)
        if not markdown_path.exists() or not refined_files:
            return {
                "paper_id": candidate.paper_id,
                "markdown_path": str(markdown_path) if markdown_path else "",
                "merged_pages": [],
                "formula_markers_after_merge": candidate.formula_marker_count,
            }

        original_text = markdown_path.read_text(encoding="utf-8")
        base_text = self._strip_existing_merge_block(original_text).rstrip()
        refined_pages = sorted({page for start, end, _path in refined_files for page in range(start, end + 1)})
        marker_pages = candidate.pages
        if marker_pages and set(marker_pages).issubset(set(refined_pages)):
            base_text = base_text.replace(f"<!-- {FORMULA_MARKER} -->", FORMULA_REFINED_POINTER)
            base_text = base_text.replace(FORMULA_MARKER, "formula-refined-see-merged-section")

        merge_block = self._build_merge_block(refined_files)
        merged_text = f"{base_text}\n\n{merge_block}\n"
        markdown_path.write_text(merged_text, encoding="utf-8")
        report = {
            "paper_id": candidate.paper_id,
            "markdown_path": str(markdown_path),
            "merged_pages": refined_pages,
            "refined_file_count": len(refined_files),
            "formula_markers_after_merge": merged_text.count(FORMULA_MARKER),
        }
        self._update_merge_report(paper_dir, report)
        return report

    @staticmethod
    def _refined_page_files(paper_dir: Path) -> list[tuple[int, int, Path]]:
        refined_files: list[tuple[int, int, Path]] = []
        if not paper_dir.exists():
            return refined_files
        for path in paper_dir.glob("pages_*_*.md"):
            match = re.fullmatch(r"pages_(\d+)_(\d+)", path.stem)
            if not match:
                continue
            refined_files.append((int(match.group(1)), int(match.group(2)), path))
        return sorted(refined_files, key=lambda item: (item[0], item[1], item[2].name))

    @staticmethod
    def _strip_existing_merge_block(text: str) -> str:
        pattern = rf"\n*{re.escape(MERGE_BLOCK_START)}.*?{re.escape(MERGE_BLOCK_END)}\s*"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    @staticmethod
    def _build_merge_block(refined_files: list[tuple[int, int, Path]]) -> str:
        blocks = [
            MERGE_BLOCK_START,
            "## Formula Refinement Merge",
            "",
            "The following blocks were merged from Docling formula enrichment results.",
        ]
        for start, end, path in refined_files:
            page_label = f"{start}" if start == end else f"{start}-{end}"
            content = path.read_text(encoding="utf-8").strip()
            blocks.extend(
                [
                    "",
                    f"### Refined Formula Pages {page_label}",
                    "",
                    content,
                ]
            )
        blocks.extend(["", MERGE_BLOCK_END])
        return "\n".join(blocks)

    @staticmethod
    def _update_merge_report(paper_dir: Path, merge_report: dict) -> None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        report_path = paper_dir / "report.json"
        report = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}
        report.update(
            {
                "merged_markdown_path": merge_report.get("markdown_path", ""),
                "merged_pages": merge_report.get("merged_pages", []),
                "formula_markers_after_merge": merge_report.get("formula_markers_after_merge", 0),
            }
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_max_pages(self, max_pages: int | None) -> int:
        return self.settings.formula_refinement_max_pages if max_pages is None else max_pages

    @staticmethod
    def _selected_pages(pages: list[int], max_pages: int) -> list[int]:
        selected = pages or [1]
        if max_pages <= 0:
            return selected
        return selected[:max_pages]

    def _pages_to_refine(
        self,
        workspace_id: str,
        candidate: FormulaRefinementCandidate,
        max_pages: int,
    ) -> list[int]:
        pages = self._selected_pages(candidate.pages, max_pages)
        paper_dir = self.refinement_root / workspace_id / candidate.paper_id
        existing_pages = self._existing_refined_pages(paper_dir)
        return [page for page in pages if page not in existing_pages]

    @staticmethod
    def _existing_refined_pages(paper_dir: Path) -> set[int]:
        pages: set[int] = set()
        if not paper_dir.exists():
            return pages
        for path in paper_dir.glob("pages_*_*.md"):
            match = re.fullmatch(r"pages_(\d+)_(\d+)", path.stem)
            if not match:
                continue
            start, end = int(match.group(1)), int(match.group(2))
            pages.update(range(start, end + 1))
        return pages

    def _convert_page_range(self, pdf_path: Path, start_page: int, end_page: int) -> str:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ImportError("Docling is not installed. Reinstall project dependencies to refine formulas.") from exc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_formula_enrichment = True
        pipeline_options.document_timeout = self.settings.formula_refinement_timeout
        if self.settings.docling_artifacts_path:
            pipeline_options.artifacts_path = str(self.settings.docling_artifacts_path)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        result = converter.convert(str(pdf_path), page_range=(start_page, end_page))
        return clean_markdown_text(result.document.export_to_markdown())

    def _convert_page_range_with_progress(
        self,
        pdf_path: Path,
        start_page: int,
        end_page: int,
        page_index: int,
        total_pages: int,
        progress_callback=None,
    ) -> str:
        if not progress_callback:
            return self._convert_page_range(pdf_path, start_page, end_page)

        total = max(total_pages, 1)
        started_at = time.monotonic()
        heartbeat_seconds = 1.0
        page_label = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._convert_page_range, pdf_path, start_page, end_page)
            while not future.done():
                elapsed = time.monotonic() - started_at
                within_page = min(0.78, 0.20 + 0.58 * (elapsed / max(self.settings.formula_refinement_timeout, 1)))
                progress_callback(
                    min(0.95, self._progress_for_page(page_index, total, within_page)),
                    f"Refining formula page {page_label} ({int(elapsed)}s elapsed)",
                )
                time.sleep(heartbeat_seconds)
            return future.result()

    @staticmethod
    def _progress_for_page(page_index: int, total_pages: int, within_page: float) -> float:
        total = max(total_pages, 1)
        clamped_index = min(max(page_index, 0), total - 1)
        clamped_within_page = min(max(within_page, 0.0), 1.0)
        return 0.05 + 0.9 * ((clamped_index + clamped_within_page) / total)

    def _pages_with_formula_markers(self, markdown_text: str, pdf_path: Path | None = None) -> list[int]:
        pages_from_lines: list[int] = []
        current_page = 1
        for line in markdown_text.splitlines():
            stripped = line.strip()
            page_match = re.match(r"(?i)^#*\s*(?:page\s+)?(\d{1,4})\s*$", stripped)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            if FORMULA_MARKER in line:
                pages_from_lines.append(current_page)
        if pages_from_lines and not (pdf_path and pdf_path.exists()):
            return self._unique_sorted_pages(pages_from_lines)

        sections = self.rag_system.parser._extract_sections(markdown_text)
        if pdf_path and pdf_path.exists():
            sections = self.rag_system.parser._assign_pdf_page_ranges(sections, pdf_path)
        pages: list[int] = []
        for section in sections:
            if FORMULA_MARKER not in section.content:
                continue
            start = max(int(section.page_start or 1), 1)
            end = max(int(section.page_end or start), start)
            pages.extend(range(start, end + 1))
        return self._unique_sorted_pages(pages)

    @staticmethod
    def _unique_sorted_pages(pages: Iterable[int]) -> list[int]:
        return sorted({page for page in pages if page >= 1})

    @staticmethod
    def _contiguous_ranges(pages: Iterable[int]) -> list[tuple[int, int]]:
        sorted_pages = FormulaRefinementService._unique_sorted_pages(pages)
        if not sorted_pages:
            return []
        ranges = []
        start = previous = sorted_pages[0]
        for page in sorted_pages[1:]:
            if page == previous + 1:
                previous = page
                continue
            ranges.append((start, previous))
            start = previous = page
        ranges.append((start, previous))
        return ranges
