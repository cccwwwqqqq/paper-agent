from agentic_rag.document_chunker import DocumentChuncker as _DocumentChuncker
from agentic_rag.parsers import ParsedDocument
from agentic_rag.settings import get_settings


class DocumentChuncker(_DocumentChuncker):
    def __init__(self):
        super().__init__(get_settings())


__all__ = ["DocumentChuncker", "ParsedDocument"]
