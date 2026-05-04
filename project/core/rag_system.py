from agentic_rag.bootstrap import create_rag_system
from agentic_rag.settings import get_settings


class RAGSystem:
    def __init__(self, collection_name: str | None = None, enable_reflection: bool = True):
        settings = get_settings()
        self._collection_name = collection_name or settings.child_collection
        self._enable_reflection = enable_reflection
        self._delegate = None

    def initialize(self):
        self._delegate = create_rag_system(
            settings=get_settings(),
            collection_name=self._collection_name,
            enable_reflection=self._enable_reflection,
            initialize=True,
        )
        return self

    def __getattr__(self, item):
        if self._delegate is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been initialized yet. Call initialize() before accessing '{item}'."
            )
        return getattr(self._delegate, item)
