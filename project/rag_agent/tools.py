from agentic_rag.agents.tools import ToolFactory as _ToolFactory
from agentic_rag.settings import get_settings


class ToolFactory(_ToolFactory):
    def __init__(self, collection, vector_db, parent_store_manager, workspace_memory, context_provider):
        super().__init__(
            settings=get_settings(),
            collection=collection,
            vector_db=vector_db,
            parent_store_manager=parent_store_manager,
            workspace_memory=workspace_memory,
            context_provider=context_provider,
        )
