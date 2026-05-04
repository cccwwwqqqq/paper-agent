from __future__ import annotations

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from agentic_rag.settings import Settings


class VectorStoreManager:
    __client: QdrantClient

    def __init__(self, settings: Settings, dense_embeddings=None):
        self.settings = settings
        self.__client = QdrantClient(path=str(settings.qdrant_db_path))
        self.__dense_embeddings = dense_embeddings
        self.__sparse_embeddings = FastEmbedSparse(model_name=settings.sparse_model)

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=len(self.__dense_embeddings.embed_query("test")),
                    distance=qmodels.Distance.COSINE,
                ),
                sparse_vectors_config={self.settings.sparse_vector_name: qmodels.SparseVectorParams()},
            )
            print(f"Collection created: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as exc:
            print(f"Warning: could not delete collection {collection_name}: {exc}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.__client,
            collection_name=collection_name,
            embedding=self.__dense_embeddings,
            sparse_embedding=self.__sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=self.settings.sparse_vector_name,
        )

    def get_filter(self, workspace_id: str, paper_id: str | None = None, section: str | None = None):
        conditions = [
            qmodels.FieldCondition(
                key="metadata.workspace_id",
                match=qmodels.MatchValue(value=workspace_id),
            )
        ]
        if paper_id:
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.paper_id",
                    match=qmodels.MatchValue(value=paper_id),
                )
            )
        if section:
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.section",
                    match=qmodels.MatchValue(value=section),
                )
            )
        return qmodels.Filter(must=conditions)

    def delete_workspace_points(self, collection_name: str, workspace_id: str):
        try:
            self.__client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(filter=self.get_filter(workspace_id=workspace_id)),
            )
        except Exception as exc:
            print(f"Warning: could not delete workspace points for {workspace_id}: {exc}")

