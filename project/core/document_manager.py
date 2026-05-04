from agentic_rag.services.ingestion_service import IngestionService


class DocumentManager(IngestionService):
    def __init__(self, rag_system):
        super().__init__(rag_system, rag_system.settings)
