from __future__ import annotations

import shutil
from pathlib import Path

from agentic_rag.settings import Settings
from agentic_rag.utils import clear_directory_contents, resolve_paper_id


class IngestionService:
    def __init__(self, rag_system, settings: Settings):
        self.rag_system = rag_system
        self.settings = settings
        self.markdown_dir = Path(settings.markdown_dir)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.source_docs_dir = Path(settings.source_docs_dir)
        self.source_docs_dir.mkdir(parents=True, exist_ok=True)
        self.last_errors: list[str] = []

    def _workspace_dir(self, workspace_id: str) -> Path:
        workspace_dir = self.markdown_dir / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    def _workspace_source_dir(self, workspace_id: str) -> Path:
        source_dir = self.source_docs_dir / workspace_id
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir

    def _save_source_document(self, doc_path: Path, workspace_id: str, paper_id: str) -> Path:
        source_dir = self._workspace_source_dir(workspace_id)
        source_path = source_dir / f"{paper_id}{doc_path.suffix.lower()}"
        if doc_path.resolve() != source_path.resolve():
            shutil.copy2(doc_path, source_path)
        return source_path

    def add_documents(self, document_paths, workspace_id: str, progress_callback=None):
        if not document_paths:
            return 0, 0

        workspace_id = workspace_id.strip() or self.settings.default_workspace_id
        workspace_dir = self._workspace_dir(workspace_id)

        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]

        if not document_paths:
            return 0, 0

        added = 0
        skipped = 0
        self.last_errors = []
        indexed_paper_ids = {
            paper.get("paper_id")
            for paper in self.rag_system.workspace_memory.list_papers(workspace_id)
            if paper.get("paper_id")
        }

        for index, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((index + 1) / len(document_paths), f"Processing {Path(doc_path).name}")

            paper_id = resolve_paper_id(Path(doc_path).stem)
            md_path = workspace_dir / f"{paper_id}.md"

            if md_path.exists() and paper_id in indexed_paper_ids:
                skipped += 1
                continue

            try:
                doc_path_obj = Path(doc_path)
                source_path = self._save_source_document(doc_path_obj, workspace_id, paper_id)
                parsed_document = self.rag_system.parser.parse(doc_path_obj, workspace_dir)
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(parsed_document, workspace_id)

                if not child_chunks:
                    skipped += 1
                    continue

                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(workspace_id, parsed_document.paper_id, parent_chunks)
                self.rag_system.workspace_memory.register_paper(
                    workspace_id,
                    {
                        "paper_id": parsed_document.paper_id,
                        "title": parsed_document.title,
                        "source_name": parsed_document.source_name,
                        "source_path": str(source_path),
                        "markdown_path": str(parsed_document.markdown_path),
                        "sections": [section.heading for section in parsed_document.sections],
                    },
                )
                added += 1
            except Exception as exc:
                error_message = f"{Path(doc_path).name}: {exc}"
                self.last_errors.append(error_message)
                print(f"Error processing {doc_path}: {exc}")
                skipped += 1

        return added, skipped

    def get_markdown_files(self, workspace_id: str):
        workspace_id = workspace_id.strip() or self.settings.default_workspace_id
        workspace_dir = self.markdown_dir / workspace_id
        if not workspace_dir.exists():
            return []
        papers = self.rag_system.workspace_memory.list_papers(workspace_id)
        if papers:
            return [paper.get("source_name", f"{paper['paper_id']}.pdf") for paper in papers]
        return sorted([p.name.replace(".md", ".pdf") for p in workspace_dir.glob("*.md")])

    def clear_all(self, workspace_id: str):
        workspace_id = workspace_id.strip() or self.settings.default_workspace_id
        workspace_dir = self._workspace_dir(workspace_id)
        source_dir = self._workspace_source_dir(workspace_id)
        clear_directory_contents(workspace_dir)
        clear_directory_contents(source_dir)
        clear_directory_contents(Path(self.settings.formula_refinement_path) / workspace_id)
        self.rag_system.parent_store.clear_store(workspace_id)
        self.rag_system.workspace_memory.clear_workspace(workspace_id)
        self.rag_system.vector_db.delete_workspace_points(self.rag_system.collection_name, workspace_id)


DocumentManager = IngestionService
