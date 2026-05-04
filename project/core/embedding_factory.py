from agentic_rag.models.embedding_factory import create_dense_embeddings as _create_dense_embeddings
from agentic_rag.settings import get_settings


def create_dense_embeddings():
    return _create_dense_embeddings(get_settings())
