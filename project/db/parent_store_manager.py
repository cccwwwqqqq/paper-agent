from pathlib import Path

from agentic_rag.settings import get_settings
from agentic_rag.storage.parent_store import ParentStoreManager as _ParentStoreManager


class ParentStoreManager(_ParentStoreManager):
    def __init__(self, store_path: str | Path | None = None):
        super().__init__(settings=get_settings(), store_path=store_path)
