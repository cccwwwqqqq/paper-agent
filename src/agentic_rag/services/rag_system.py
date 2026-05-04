from __future__ import annotations

import uuid
from pathlib import Path

from langchain_qdrant.qdrant import QdrantVectorStoreError

from agentic_rag.agents.graph import create_agent_graph
from agentic_rag.agents.tools import ToolFactory
from agentic_rag.document_chunker import DocumentChuncker
from agentic_rag.observability import Observability
from agentic_rag.parsers import create_pdf_parser
from agentic_rag.services.evidence_retriever import EvidenceRetriever
from agentic_rag.settings import Settings
from agentic_rag.storage.parent_store import ParentStoreManager
from agentic_rag.storage.vector_store import VectorStoreManager
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore


class RAGSystem:
    def __init__(
        self,
        settings: Settings,
        llm,
        reranker,
        vector_db: VectorStoreManager,
        parent_store: ParentStoreManager,
        chunker: DocumentChuncker,
        parser,
        workspace_memory: WorkspaceMemoryStore,
        observability: Observability,
        collection_name: str | None = None,
        enable_reflection: bool = True,
    ):
        self.settings = settings
        self.collection_name = collection_name or settings.child_collection
        self.enable_reflection = enable_reflection
        self.llm = llm
        self.reranker = reranker
        self.vector_db = vector_db
        self.parent_store = parent_store
        self.chunker = chunker
        self.parser = parser
        self.workspace_memory = workspace_memory
        self.observability = observability
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = settings.graph_recursion_limit
        self._thread_workspace_id = settings.default_workspace_id
        self.active_context = {
            "workspace_context": {
                "workspace_id": settings.default_workspace_id,
                "focus_paper_id": None,
                "selected_focus_paper_id": None,
                "current_dialogue_paper_id": None,
                "knowledge_scope": settings.knowledge_scope,
                "intent_type": "general_retrieval",
            }
        }

    def set_workspace_context(
        self,
        workspace_id: str,
        focus_paper_id: str | None = None,
        intent_type: str = "general_retrieval",
    ):
        normalized_workspace = workspace_id.strip() or self.settings.default_workspace_id
        previous_context = self.active_context.get("workspace_context", {}) or {}
        if normalized_workspace != self._thread_workspace_id:
            self.reset_thread()
            self._thread_workspace_id = normalized_workspace
            resolved_focus_paper_id = focus_paper_id
            current_dialogue_paper_id = None
        else:
            previous_selected_focus = previous_context.get("selected_focus_paper_id")
            previous_resolved_focus = previous_context.get("focus_paper_id")
            previous_dialogue_focus = previous_context.get("current_dialogue_paper_id")
            if focus_paper_id != previous_selected_focus:
                resolved_focus_paper_id = focus_paper_id
            else:
                resolved_focus_paper_id = previous_resolved_focus
            current_dialogue_paper_id = previous_dialogue_focus
        self.active_context = {
            "workspace_context": {
                "workspace_id": normalized_workspace,
                "focus_paper_id": resolved_focus_paper_id,
                "selected_focus_paper_id": focus_paper_id,
                "current_dialogue_paper_id": current_dialogue_paper_id,
                "knowledge_scope": self.settings.knowledge_scope,
                "intent_type": intent_type,
            }
        }
        self.workspace_memory.ensure_workspace(self.active_context["workspace_context"]["workspace_id"])

    def get_workspace_context(self):
        return self.active_context

    def sync_workspace_context(self, workspace_context: dict | None):
        if not workspace_context:
            return
        previous_context = self.active_context.get("workspace_context", {}) or {}
        merged_context = {
            "workspace_id": workspace_context.get("workspace_id", previous_context.get("workspace_id", self.settings.default_workspace_id)),
            "focus_paper_id": workspace_context.get("focus_paper_id", previous_context.get("focus_paper_id")),
            "selected_focus_paper_id": workspace_context.get(
                "selected_focus_paper_id",
                previous_context.get("selected_focus_paper_id"),
            ),
            "current_dialogue_paper_id": workspace_context.get(
                "current_dialogue_paper_id",
                previous_context.get("current_dialogue_paper_id"),
            ),
            "knowledge_scope": workspace_context.get("knowledge_scope", previous_context.get("knowledge_scope", self.settings.knowledge_scope)),
            "intent_type": workspace_context.get("intent_type", previous_context.get("intent_type", "general_retrieval")),
        }
        self.active_context = {"workspace_context": merged_context}

    def _rebuild_indexes_from_markdown(self, collection):
        markdown_root = Path(self.settings.markdown_dir)
        if not markdown_root.exists():
            return

        md_files = list(markdown_root.rglob("*.md"))
        if not md_files:
            return

        print("Rebuilding workspace indexes from markdown files...")
        for md_path in md_files:
            workspace_id = md_path.parent.name if md_path.parent != markdown_root else self.settings.default_workspace_id
            parsed_document = self.parser.parse(md_path, md_path.parent)
            parent_chunks, child_chunks = self.chunker.create_chunks_single(parsed_document, workspace_id)
            if not child_chunks:
                continue

            collection.add_documents(child_chunks)
            self.parent_store.save_many(workspace_id, parsed_document.paper_id, parent_chunks)
            self.workspace_memory.register_paper(
                workspace_id,
                {
                    "paper_id": parsed_document.paper_id,
                    "title": parsed_document.title,
                    "source_name": parsed_document.source_name,
                    "markdown_path": str(parsed_document.markdown_path),
                    "sections": [section.heading for section in parsed_document.sections],
                },
            )

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        try:
            collection = self.vector_db.get_collection(self.collection_name)
        except QdrantVectorStoreError as exc:
            message = str(exc)
            if "Selected embeddings are" not in message:
                raise

            print("Embedding dimension changed. Recreating Qdrant collection and rebuilding indexes...")
            self.vector_db.delete_collection(self.collection_name)
            self.vector_db.create_collection(self.collection_name)
            collection = self.vector_db.get_collection(self.collection_name)
            self._rebuild_indexes_from_markdown(collection)

        evidence_retriever = EvidenceRetriever(
            settings=self.settings,
            collection=collection,
            vector_db=self.vector_db,
            parent_store=self.parent_store,
            reranker=self.reranker,
        )
        tools = ToolFactory(
            settings=self.settings,
            evidence_retriever=evidence_retriever,
            parent_store_manager=self.parent_store,
            workspace_memory=self.workspace_memory,
            context_provider=self.get_workspace_context,
        ).create_tools()
        self.agent_graph = create_agent_graph(
            llm=self.llm,
            tools_list=tools,
            collection=collection,
            vector_db=self.vector_db,
            parent_store=self.parent_store,
            evidence_retriever=evidence_retriever,
            workspace_memory=self.workspace_memory,
            settings=self.settings,
            enable_reflection=self.enable_reflection,
        )

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as exc:
            print(f"Warning: Could not delete thread {self.thread_id}: {exc}")
        self.thread_id = str(uuid.uuid4())
