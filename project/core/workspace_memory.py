from pathlib import Path

from agentic_rag.settings import get_settings
from agentic_rag.storage.workspace_memory import WorkspaceMemoryStore as _WorkspaceMemoryStore


class WorkspaceMemoryStore(_WorkspaceMemoryStore):
    def __init__(self, root_path: str | Path | None = None):
        super().__init__(settings=get_settings(), root_path=root_path)
