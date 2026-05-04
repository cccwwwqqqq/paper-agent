from agentic_rag.models.llm_factory import create_llm as _create_llm
from agentic_rag.settings import get_settings


def create_llm():
    return _create_llm(get_settings())
