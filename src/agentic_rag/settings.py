from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_legacy_data_path(root_dir: Path, data_dir: Path, folder_name: str) -> Path:
    default_path = data_dir / folder_name
    legacy_candidates = [root_dir / folder_name]
    if not folder_name.startswith("."):
        legacy_candidates.append(root_dir / f".{folder_name}")
    legacy_path = next((candidate for candidate in legacy_candidates if candidate.exists()), legacy_candidates[0])

    if not legacy_path.exists():
        return default_path
    if not default_path.exists():
        return legacy_path

    try:
        default_has_content = any(default_path.iterdir())
    except OSError:
        default_has_content = False

    try:
        legacy_has_content = any(legacy_path.iterdir())
    except OSError:
        legacy_has_content = False

    if legacy_has_content and not default_has_content:
        return legacy_path
    return default_path


def _discover_env_file(root_dir: Path, explicit_env_file: str | Path | None) -> Path | None:
    if explicit_env_file:
        env_path = Path(explicit_env_file)
        if not env_path.is_absolute():
            env_path = root_dir / env_path
        return env_path

    root_env = root_dir / ".env"
    if root_env.exists():
        return root_env

    legacy_env = root_dir / "project" / ".env"
    if legacy_env.exists():
        return legacy_env

    return None


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    env_file: Path | None
    data_dir: Path
    assets_dir: Path
    eval_dir: Path
    markdown_dir: Path
    source_docs_dir: Path
    parent_store_path: Path
    formula_refinement_path: Path
    qdrant_db_path: Path
    workspace_memory_path: Path
    docling_artifacts_path: Path
    docling_do_ocr: bool
    docling_do_formula_enrichment: bool
    child_collection: str
    sparse_vector_name: str
    embedding_provider: str
    dense_model: str
    sparse_model: str
    embedding_model: str
    embedding_api_key: str
    embedding_base_url: str
    rerank_provider: str
    rerank_model: str
    rerank_api_key: str
    rerank_base_url: str
    rerank_top_n: int
    rerank_max_candidates: int
    llm_provider: str
    llm_model: str
    llm_temperature: float
    ollama_base_url: str
    openai_api_key: str
    openai_base_url: str
    deepseek_api_key: str
    deepseek_base_url: str
    openai_compat_api_key: str
    openai_compat_base_url: str
    default_workspace_id: str
    knowledge_scope: str
    pdf_parser_backend: str
    default_search_limit: int
    max_tool_calls: int
    max_iterations: int
    graph_recursion_limit: int
    base_token_threshold: int
    token_growth_factor: float
    child_chunk_size: int
    child_chunk_overlap: int
    min_parent_size: int
    max_parent_size: int
    formula_refinement_max_pages: int
    formula_refinement_timeout: int
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str

    @property
    def headers_to_split_on(self) -> list[tuple[str, str]]:
        return [("#", "H1"), ("##", "H2"), ("###", "H3")]


def load_settings(
    env_file: str | Path | None = None,
    root_dir: str | Path | None = None,
    *,
    use_env: bool = True,
) -> Settings:
    resolved_root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    resolved_root = resolved_root.resolve()

    discovered_env = _discover_env_file(resolved_root, env_file)
    if use_env and discovered_env and discovered_env.exists():
        load_dotenv(discovered_env, override=False)

    data_dir = resolved_root / os.environ.get("DATA_DIR", "data")

    dense_model = os.environ.get("DENSE_MODEL", "sentence-transformers/all-mpnet-base-v2")
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
    embedding_model = os.environ.get("EMBEDDING_MODEL", dense_model)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
    openai_compat_api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "")
    openai_compat_base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", "")

    docling_path_raw = os.environ.get("DOCLING_ARTIFACTS_PATH", "").strip()
    if docling_path_raw:
        docling_path = Path(docling_path_raw)
        if not docling_path.is_absolute():
            docling_path = resolved_root / docling_path
    else:
        docling_path = _resolve_legacy_data_path(resolved_root, data_dir, "docling_artifacts")

    return Settings(
        root_dir=resolved_root,
        env_file=discovered_env,
        data_dir=data_dir,
        assets_dir=resolved_root / "assets",
        eval_dir=resolved_root / "eval",
        markdown_dir=_resolve_legacy_data_path(resolved_root, data_dir, "markdown_docs"),
        source_docs_dir=_resolve_legacy_data_path(resolved_root, data_dir, "source_docs"),
        parent_store_path=_resolve_legacy_data_path(resolved_root, data_dir, "parent_store"),
        formula_refinement_path=_resolve_legacy_data_path(resolved_root, data_dir, "formula_refinements"),
        qdrant_db_path=_resolve_legacy_data_path(resolved_root, data_dir, "qdrant_db"),
        workspace_memory_path=_resolve_legacy_data_path(resolved_root, data_dir, "workspace_memory"),
        docling_artifacts_path=docling_path,
        docling_do_ocr=_as_bool(os.environ.get("DOCLING_DO_OCR"), default=False),
        docling_do_formula_enrichment=_as_bool(os.environ.get("DOCLING_DO_FORMULA_ENRICHMENT"), default=False),
        child_collection="document_child_chunks",
        sparse_vector_name="sparse",
        embedding_provider=embedding_provider,
        dense_model=dense_model,
        sparse_model="Qdrant/bm25",
        embedding_model=embedding_model,
        embedding_api_key=os.environ.get(
            "EMBEDDING_API_KEY",
            openai_compat_api_key or openai_api_key,
        ),
        embedding_base_url=os.environ.get(
            "EMBEDDING_BASE_URL",
            openai_compat_base_url or openai_base_url,
        ),
        rerank_provider=os.environ.get("RERANK_PROVIDER", "siliconflow").lower(),
        rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_api_key=os.environ.get(
            "RERANK_API_KEY",
            openai_compat_api_key or openai_api_key,
        ),
        rerank_base_url=os.environ.get("RERANK_BASE_URL", "https://api.siliconflow.cn/v1/rerank"),
        rerank_top_n=int(os.environ.get("RERANK_TOP_N", "6")),
        rerank_max_candidates=int(os.environ.get("RERANK_MAX_CANDIDATES", "24")),
        llm_provider=os.environ.get("LLM_PROVIDER", "ollama").lower(),
        llm_model=os.environ.get("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        openai_compat_api_key=openai_compat_api_key,
        openai_compat_base_url=openai_compat_base_url,
        default_workspace_id=os.environ.get("DEFAULT_WORKSPACE_ID", "default").strip() or "default",
        knowledge_scope=os.environ.get("KNOWLEDGE_SCOPE", "workspace_documents"),
        pdf_parser_backend=os.environ.get("PDF_PARSER_BACKEND", "pymupdf4llm").lower(),
        default_search_limit=int(os.environ.get("DEFAULT_SEARCH_LIMIT", "6")),
        max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", "8")),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "10")),
        graph_recursion_limit=int(os.environ.get("GRAPH_RECURSION_LIMIT", "50")),
        base_token_threshold=int(os.environ.get("BASE_TOKEN_THRESHOLD", "2000")),
        token_growth_factor=float(os.environ.get("TOKEN_GROWTH_FACTOR", "0.9")),
        child_chunk_size=int(os.environ.get("CHILD_CHUNK_SIZE", "500")),
        child_chunk_overlap=int(os.environ.get("CHILD_CHUNK_OVERLAP", "100")),
        min_parent_size=int(os.environ.get("MIN_PARENT_SIZE", "2000")),
        max_parent_size=int(os.environ.get("MAX_PARENT_SIZE", "4000")),
        formula_refinement_max_pages=int(os.environ.get("FORMULA_REFINEMENT_MAX_PAGES", "0")),
        formula_refinement_timeout=int(os.environ.get("FORMULA_REFINEMENT_TIMEOUT", "180")),
        langfuse_enabled=_as_bool(os.environ.get("LANGFUSE_ENABLED"), default=False),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        langfuse_base_url=os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
