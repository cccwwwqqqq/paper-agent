from __future__ import annotations

from agentic_rag.settings import Settings


def create_dense_embeddings(settings: Settings):
    provider = settings.embedding_provider

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if provider in {"openai", "openai_compatible", "siliconflow"}:
        if not settings.embedding_api_key:
            raise ValueError(
                "Embedding API key is missing. Set EMBEDDING_API_KEY or OPENAI_COMPAT_API_KEY / OPENAI_API_KEY."
            )

        from langchain_openai import OpenAIEmbeddings

        kwargs = {
            "model": settings.embedding_model,
            "api_key": settings.embedding_api_key,
        }
        if settings.embedding_base_url:
            kwargs["base_url"] = settings.embedding_base_url
        return OpenAIEmbeddings(**kwargs)

    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER '{}'. Use one of: huggingface, openai, openai_compatible, siliconflow.".format(
            provider
        )
    )
