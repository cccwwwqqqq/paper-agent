from agentic_rag.parsers.adapters import DoclingParserAdapter, ParsedDocument, ParsedSection, PdfParserAdapter
from agentic_rag.parsers.adapters import PymuPdf4LlmParserAdapter
from agentic_rag.settings import get_settings


class MinerUParserAdapter(PdfParserAdapter):
    def parse(self, pdf_path, output_dir):
        raise NotImplementedError("MinerU parser is not enabled in the compatibility layer.")


def create_pdf_parser() -> PdfParserAdapter:
    from agentic_rag.parsers import create_pdf_parser as _create_pdf_parser

    return _create_pdf_parser(get_settings())


__all__ = [
    "DoclingParserAdapter",
    "MinerUParserAdapter",
    "ParsedDocument",
    "ParsedSection",
    "PdfParserAdapter",
    "PymuPdf4LlmParserAdapter",
    "create_pdf_parser",
]
