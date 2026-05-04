"""Legacy compatibility config module.

Prefer using `agentic_rag.settings.Settings` in new code.
"""

from agentic_rag.settings import get_settings

_settings = get_settings()

MARKDOWN_DIR = str(_settings.markdown_dir)
PARENT_STORE_PATH = str(_settings.parent_store_path)
QDRANT_DB_PATH = str(_settings.qdrant_db_path)
WORKSPACE_MEMORY_PATH = str(_settings.workspace_memory_path)
CHILD_COLLECTION = _settings.child_collection
SPARSE_VECTOR_NAME = _settings.sparse_vector_name
EMBEDDING_PROVIDER = _settings.embedding_provider
DENSE_MODEL = _settings.dense_model
SPARSE_MODEL = _settings.sparse_model
EMBEDDING_MODEL = _settings.embedding_model
EMBEDDING_API_KEY = _settings.embedding_api_key
EMBEDDING_BASE_URL = _settings.embedding_base_url
LLM_PROVIDER = _settings.llm_provider
LLM_MODEL = _settings.llm_model
LLM_TEMPERATURE = _settings.llm_temperature
OLLAMA_BASE_URL = _settings.ollama_base_url
OPENAI_API_KEY = _settings.openai_api_key
OPENAI_BASE_URL = _settings.openai_base_url
DEEPSEEK_API_KEY = _settings.deepseek_api_key
DEEPSEEK_BASE_URL = _settings.deepseek_base_url
OPENAI_COMPAT_API_KEY = _settings.openai_compat_api_key
OPENAI_COMPAT_BASE_URL = _settings.openai_compat_base_url
DEFAULT_WORKSPACE_ID = _settings.default_workspace_id
KNOWLEDGE_SCOPE = _settings.knowledge_scope
PDF_PARSER_BACKEND = _settings.pdf_parser_backend
DOCLING_ARTIFACTS_PATH = str(_settings.docling_artifacts_path)
DOCLING_DO_FORMULA_ENRICHMENT = _settings.docling_do_formula_enrichment
MAX_TOOL_CALLS = _settings.max_tool_calls
MAX_ITERATIONS = _settings.max_iterations
GRAPH_RECURSION_LIMIT = _settings.graph_recursion_limit
BASE_TOKEN_THRESHOLD = _settings.base_token_threshold
TOKEN_GROWTH_FACTOR = _settings.token_growth_factor
CHILD_CHUNK_SIZE = _settings.child_chunk_size
CHILD_CHUNK_OVERLAP = _settings.child_chunk_overlap
MIN_PARENT_SIZE = _settings.min_parent_size
MAX_PARENT_SIZE = _settings.max_parent_size
HEADERS_TO_SPLIT_ON = _settings.headers_to_split_on
DEFAULT_SEARCH_LIMIT = _settings.default_search_limit
LANGFUSE_ENABLED = _settings.langfuse_enabled
LANGFUSE_PUBLIC_KEY = _settings.langfuse_public_key
LANGFUSE_SECRET_KEY = _settings.langfuse_secret_key
LANGFUSE_BASE_URL = _settings.langfuse_base_url
