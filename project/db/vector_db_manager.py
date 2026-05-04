from agentic_rag.models.embedding_factory import create_dense_embeddings
from agentic_rag.settings import get_settings
from agentic_rag.storage.vector_store import VectorStoreManager as _VectorStoreManager


class VectorDbManager(_VectorStoreManager):
    def __init__(self):
        settings = get_settings()
        super().__init__(settings=settings, dense_embeddings=create_dense_embeddings(settings))
