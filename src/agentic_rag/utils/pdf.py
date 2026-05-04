from __future__ import annotations

import glob
import os
from pathlib import Path

from agentic_rag.settings import Settings
from agentic_rag.utils.text import clean_markdown_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pdf_to_markdown(pdf_path, output_dir, output_stem: str | None = None):
    import pymupdf
    import pymupdf.layout  # noqa: F401 - imported for pymupdf4llm compatibility
    import pymupdf4llm

    doc = pymupdf.open(pdf_path)
    markdown = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,
        ignore_images=True,
        write_images=False,
        image_path=None,
    )
    cleaned = markdown.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    cleaned = clean_markdown_text(cleaned)
    safe_stem = output_stem or Path(doc.name).stem
    output_path = Path(output_dir) / safe_stem
    markdown_path = Path(output_path).with_suffix(".md")
    markdown_path.write_bytes(cleaned.encode("utf-8"))
    return markdown_path


def pdfs_to_markdowns(path_pattern: str, settings: Settings, overwrite: bool = False) -> None:
    output_dir = Path(settings.markdown_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)
