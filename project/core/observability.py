from agentic_rag.observability import Observability as _Observability
from agentic_rag.settings import get_settings


class Observability(_Observability):
    def __init__(self):
        super().__init__(get_settings())
