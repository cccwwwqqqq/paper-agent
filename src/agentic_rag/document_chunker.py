from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.parsers import ParsedDocument
from agentic_rag.settings import Settings
from agentic_rag.utils.markdown_chunking import is_formula_like_document, split_formula_aware_documents


class DocumentChuncker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
        )
        self.__min_parent_size = settings.min_parent_size
        self.__max_parent_size = settings.max_parent_size

    def create_chunks(self, path_dir: str | Path | None = None, workspace_id: str | None = None):
        target_path = str(path_dir or self.settings.markdown_dir)
        resolved_workspace = workspace_id or self.settings.default_workspace_id
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(target_path, "*.md"))):
            doc_path = Path(doc_path_str)
            parsed_document = ParsedDocument(
                paper_id=doc_path.stem,
                source_name=f"{doc_path.stem}.pdf",
                markdown_path=doc_path,
                title=doc_path.stem,
            )
            parent_chunks, child_chunks = self.create_chunks_single(parsed_document, resolved_workspace)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)

        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, parsed_document: ParsedDocument, workspace_id: str):
        cleaned_parents = self._build_parent_chunks(parsed_document, workspace_id)
        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(
            all_parent_chunks,
            all_child_chunks,
            cleaned_parents,
            workspace_id,
            parsed_document.paper_id,
            parsed_document.source_name,
        )
        return all_parent_chunks, all_child_chunks

    def _build_parent_chunks(self, parsed_document: ParsedDocument, workspace_id: str):
        parents: list[Document] = []

        if not parsed_document.sections:
            text = parsed_document.markdown_path.read_text(encoding="utf-8")
            parsed_document.sections = []
            if text.strip():
                parsed_document.sections.append(
                    type(
                        "Section",
                        (),
                        {
                            "heading": parsed_document.title,
                            "content": text,
                            "section_type": "section",
                            "page_start": 1,
                            "page_end": 1,
                            "block_type": "paragraph",
                        },
                    )()
                )

        for section in parsed_document.sections:
            content = section.content.strip()
            if not content:
                continue

            header = f"# {section.heading}\n\n" if section.heading else ""
            section_norm = self._normalize_section_name(section.heading)
            content_type = self._detect_content_type(section.heading, content)
            equation_dense = self._is_equation_dense(content)
            table_caption = self._extract_caption(section.heading, content, target="table")
            figure_caption = self._extract_caption(section.heading, content, target="figure")
            document = Document(
                page_content=f"{header}{content}".strip(),
                metadata={
                    "workspace_id": workspace_id,
                    "paper_id": parsed_document.paper_id,
                    "source": parsed_document.source_name,
                    "title": parsed_document.title,
                    "section": section.heading,
                    "section_norm": section_norm,
                    "section_type": section.section_type,
                    "block_type": section.block_type,
                    "content_type": content_type,
                    "equation_dense": equation_dense,
                    "table_caption": table_caption,
                    "figure_caption": figure_caption,
                    "page_start": section.page_start or 1,
                    "page_end": section.page_end or section.page_start or 1,
                },
            )
            parents.extend(self._normalize_parent(document))

        return parents

    def _normalize_parent(self, parent_doc: Document):
        if len(parent_doc.page_content) <= self.__max_parent_size:
            return [parent_doc]

        if is_formula_like_document(parent_doc):
            return split_formula_aware_documents(
                parent_doc,
                chunk_size=self.__max_parent_size,
                chunk_overlap=self.settings.child_chunk_overlap,
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.__max_parent_size,
            chunk_overlap=self.settings.child_chunk_overlap,
        )
        split_docs = splitter.split_documents([parent_doc])
        for split_doc in split_docs:
            split_doc.metadata = dict(parent_doc.metadata)
        return split_docs

    @staticmethod
    def _normalize_section_name(section_name: str) -> str:
        normalized = re.sub(r"[_*`#]", " ", str(section_name or "").lower())
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _detect_content_type(section_name: str, content: str) -> str:
        heading = str(section_name or "").lower()
        body = str(content or "")
        lowered = body.lower()
        non_empty_lines = [line.strip() for line in body.splitlines() if line.strip()]

        algorithm_markers = (
            "algorithm",
            "setup",
            "keygen",
            "enc",
            "encrypt",
            "dec",
            "decrypt",
            "procedure",
            "workflow",
        )
        formula_markers = (
            "formula-not-decoded",
            "equation-not-decoded",
            "lambda",
            "bilinear",
            "rlwe",
            "q-sdh",
            "e(",
        )

        if any(token in heading for token in ("figure", "fig.")):
            return "figure"
        if any(token in heading for token in ("table",)):
            return "table"
        if any(token in heading for token in ("caption", "legend")):
            return "caption"
        if re.match(r"^\s*(fig(?:ure)?\.?\s*\d+|table\s*\d+)", body, re.IGNORECASE):
            return "caption"
        if any(token in heading for token in algorithm_markers):
            return "algorithm"
        if any(token in lowered for token in formula_markers):
            return "formula"
        if any(token in lowered for token in ("figure", "fig.")):
            return "figure"
        if any(token in lowered for token in ("table",)):
            return "table"
        if any(token in lowered for token in ("caption", "legend")):
            return "caption"
        if non_empty_lines and sum(1 for line in non_empty_lines[:6] if re.match(r"^(\d+\.|[-*]|[a-z]\))", line.lower())) >= 2:
            return "list"
        if any(token in lowered for token in algorithm_markers):
            return "algorithm"
        return "paragraph"

    @staticmethod
    def _is_equation_dense(content: str) -> bool:
        body = str(content or "")
        if not body.strip():
            return False
        formula_hits = len(
            re.findall(
                r"(formula-not-decoded|equation-not-decoded|lambda|bilinear|rlwe|q-sdh|e\(|[_^]\{|=\s*.+|[a-z]\s*\\to\s*[a-z])",
                body,
                flags=re.IGNORECASE,
            )
        )
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        symbolic_lines = sum(1 for line in lines[:10] if re.search(r"[=_^\\{}()]{2,}", line))
        return formula_hits >= 3 or symbolic_lines >= 2

    @staticmethod
    def _extract_caption(section_name: str, content: str, *, target: str) -> str:
        heading = str(section_name or "").strip()
        body = str(content or "").strip()
        prefix = "fig" if target == "figure" else "table"
        heading_match = re.search(rf"({prefix}(?:ure)?\.?\s*\d+.*)", heading, flags=re.IGNORECASE)
        if heading_match:
            return heading_match.group(1).strip()

        for line in body.splitlines()[:5]:
            stripped = line.strip()
            if re.match(rf"^{prefix}(?:ure)?\.?\s*\d+.*", stripped, flags=re.IGNORECASE):
                return stripped
        return ""

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, workspace_id, paper_id, source_name):
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{paper_id}_parent_{i}"
            p_chunk.metadata.update(
                {
                    "workspace_id": workspace_id,
                    "paper_id": paper_id,
                    "source": source_name,
                    "parent_id": parent_id,
                }
            )
            all_parent_pairs.append((parent_id, p_chunk))

            if is_formula_like_document(p_chunk):
                child_docs = split_formula_aware_documents(
                    p_chunk,
                    chunk_size=self.settings.child_chunk_size,
                    chunk_overlap=self.settings.child_chunk_overlap,
                )
            else:
                child_docs = self.__child_splitter.split_documents([p_chunk])
            for child_idx, child_doc in enumerate(child_docs):
                child_doc.metadata.update(
                    {
                        "workspace_id": workspace_id,
                        "paper_id": paper_id,
                        "source": source_name,
                        "parent_id": parent_id,
                        "child_id": f"{parent_id}_child_{child_idx}",
                        "section": p_chunk.metadata.get("section", "Unknown Section"),
                        "section_norm": p_chunk.metadata.get("section_norm", ""),
                        "block_type": p_chunk.metadata.get("block_type", "paragraph"),
                        "content_type": p_chunk.metadata.get("content_type", "paragraph"),
                        "equation_dense": p_chunk.metadata.get("equation_dense", False),
                        "table_caption": p_chunk.metadata.get("table_caption", ""),
                        "figure_caption": p_chunk.metadata.get("figure_caption", ""),
                        "page_start": p_chunk.metadata.get("page_start", 1),
                        "page_end": p_chunk.metadata.get("page_end", 1),
                        "title": p_chunk.metadata.get("title", paper_id),
                    }
                )
            all_child_chunks.extend(child_docs)
