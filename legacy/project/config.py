import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")
WORKSPACE_MEMORY_PATH = os.path.join(_BASE_DIR, "workspace_memory")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
DENSE_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-mpnet-base-v2")
SPARSE_MODEL = "Qdrant/bm25"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", DENSE_MODEL)
EMBEDDING_API_KEY = os.environ.get(
    "EMBEDDING_API_KEY",
    os.environ.get("OPENAI_COMPAT_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
)
EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL",
    os.environ.get("OPENAI_COMPAT_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")),
)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

OPENAI_COMPAT_API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "")

# --- Workspace / Parser Configuration ---
DEFAULT_WORKSPACE_ID = os.environ.get("DEFAULT_WORKSPACE_ID", "default").strip() or "default"
KNOWLEDGE_SCOPE = os.environ.get("KNOWLEDGE_SCOPE", "workspace_documents")
PDF_PARSER_BACKEND = os.environ.get("PDF_PARSER_BACKEND", "pymupdf4llm").lower()
DOCLING_ARTIFACTS_PATH = os.environ.get("DOCLING_ARTIFACTS_PATH", "").strip()

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
DEFAULT_SEARCH_LIMIT = int(os.environ.get("DEFAULT_SEARCH_LIMIT", "6"))

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
