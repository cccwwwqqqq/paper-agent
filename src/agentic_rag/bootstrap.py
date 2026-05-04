from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from agentic_rag.document_chunker import DocumentChuncker
from agentic_rag.models import create_dense_embeddings, create_llm, create_reranker
from agentic_rag.observability import Observability
from agentic_rag.parsers import create_pdf_parser
from agentic_rag.services.chat_service import ChatService
from agentic_rag.services.formula_refinement_service import FormulaRefinementService
from agentic_rag.services.ingestion_service import IngestionService
from agentic_rag.services.rag_system import RAGSystem
from agentic_rag.settings import Settings, get_settings
from agentic_rag.storage.parent_store import ParentStoreManager
from agentic_rag.storage.vector_store import VectorStoreManager
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore


@dataclass
class ApplicationRuntime:
    settings: Settings
    rag_system: RAGSystem
    chat_service: ChatService
    ingestion_service: IngestionService
    formula_refinement_service: FormulaRefinementService


def create_rag_system(
    settings: Settings | None = None,
    *,
    collection_name: str | None = None,
    enable_reflection: bool = True,
    initialize: bool = True,
) -> RAGSystem:
    resolved_settings = settings or get_settings()
    if collection_name and collection_name != resolved_settings.child_collection:
        resolved_settings = replace(resolved_settings, child_collection=collection_name)

    dense_embeddings = create_dense_embeddings(resolved_settings)
    vector_store = VectorStoreManager(resolved_settings, dense_embeddings=dense_embeddings)
    parent_store = ParentStoreManager(resolved_settings)
    chunker = DocumentChuncker(resolved_settings)
    parser = create_pdf_parser(resolved_settings)
    workspace_memory = WorkspaceMemoryStore(resolved_settings)
    observability = Observability(resolved_settings)
    llm = create_llm(resolved_settings)
    reranker = create_reranker(resolved_settings)

    rag_system = RAGSystem(
        settings=resolved_settings,
        llm=llm,
        reranker=reranker,
        vector_db=vector_store,
        parent_store=parent_store,
        chunker=chunker,
        parser=parser,
        workspace_memory=workspace_memory,
        observability=observability,
        enable_reflection=enable_reflection,
    )
    if initialize:
        rag_system.initialize()
    return rag_system


def build_runtime(settings: Settings | None = None, *, enable_reflection: bool = True) -> ApplicationRuntime:
    resolved_settings = settings or get_settings()
    rag_system = create_rag_system(resolved_settings, enable_reflection=enable_reflection, initialize=True)
    ingestion_service = IngestionService(rag_system, resolved_settings)
    formula_refinement_service = FormulaRefinementService(rag_system, resolved_settings)
    chat_service = ChatService(rag_system)

    return ApplicationRuntime(
        settings=resolved_settings,
        rag_system=rag_system,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
        formula_refinement_service=formula_refinement_service,
    )
