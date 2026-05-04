from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


_MATH_ENV_START = re.compile(r"\\begin\{(equation|align|gather|multline|split|cases)\*?\}")
_MATH_ENV_END = re.compile(r"\\end\{(equation|align|gather|multline|split|cases)\*?\}")


def is_formula_like_document(document: Document) -> bool:
    metadata = document.metadata or {}
    content_type = str(metadata.get("content_type", "")).lower()
    return bool(metadata.get("equation_dense")) or content_type in {"formula", "formula_refined"}


def split_formula_aware_documents(document: Document, *, chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunks = split_formula_aware_text(
        document.page_content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [Document(page_content=chunk, metadata=dict(document.metadata)) for chunk in chunks if chunk.strip()]


def split_formula_aware_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    blocks = _markdown_blocks(text)
    if not blocks:
        return _fallback_split(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        chunk = _join_blocks(current)
        if chunk.strip():
            chunks.append(chunk)
        current = _overlap_tail(current, chunk_overlap)

    for block in blocks:
        if not block.strip():
            continue
        if not current:
            current = [block]
            if len(_join_blocks(current)) > chunk_size:
                current = []
                chunks.extend(_split_oversized_block(block, chunk_size, chunk_overlap))
            continue

        candidate = _join_blocks([*current, block])
        if len(candidate) <= chunk_size:
            current.append(block)
            continue

        flush()
        if not current:
            current = [block]
        else:
            candidate = _join_blocks([*current, block])
            if len(candidate) <= chunk_size:
                current.append(block)
            else:
                flush()
                current = [block]

        if len(_join_blocks(current)) > chunk_size:
            oversized = current.pop()
            if current:
                flush()
            chunks.extend(_split_oversized_block(oversized, chunk_size, chunk_overlap))
            current = []

    if current:
        chunk = _join_blocks(current)
        if chunk.strip():
            chunks.append(chunk)

    return _with_heading_context(_dedupe_adjacent_chunks(chunks), blocks, chunk_size)


def _markdown_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append("\n".join(paragraph).strip())
            paragraph = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if _starts_fenced_code(stripped):
            flush_paragraph()
            block, index = _collect_until_fence(lines, index)
            blocks.append(block)
            continue

        if _starts_display_math(stripped):
            flush_paragraph()
            block, index = _collect_math_block(lines, index)
            blocks.append(block)
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            blocks.append(line.rstrip())
            index += 1
            continue

        if _looks_like_table_line(stripped):
            flush_paragraph()
            block, index = _collect_table(lines, index)
            blocks.append(block)
            continue

        paragraph.append(line.rstrip())
        index += 1

    flush_paragraph()
    return [block for block in blocks if block.strip()]


def _starts_fenced_code(stripped: str) -> bool:
    return stripped.startswith("```") or stripped.startswith("~~~")


def _collect_until_fence(lines: list[str], start: int) -> tuple[str, int]:
    opener = lines[start].strip()[:3]
    collected = [lines[start].rstrip()]
    index = start + 1
    while index < len(lines):
        collected.append(lines[index].rstrip())
        if lines[index].strip().startswith(opener):
            index += 1
            break
        index += 1
    return "\n".join(collected).strip(), index


def _starts_display_math(stripped: str) -> bool:
    return (
        stripped.startswith("$$")
        or stripped.startswith("\\[")
        or bool(_MATH_ENV_START.search(stripped))
        or stripped == "<!-- formula-not-decoded -->"
    )


def _collect_math_block(lines: list[str], start: int) -> tuple[str, int]:
    first = lines[start].strip()
    collected = [lines[start].rstrip()]
    index = start + 1

    if first == "<!-- formula-not-decoded -->":
        return "\n".join(collected).strip(), index

    if first.startswith("$$"):
        if first.count("$$") >= 2 and len(first) > 2:
            return "\n".join(collected).strip(), index
        while index < len(lines):
            collected.append(lines[index].rstrip())
            if "$$" in lines[index]:
                index += 1
                break
            index += 1
        return "\n".join(collected).strip(), index

    if first.startswith("\\["):
        if "\\]" in first:
            return "\n".join(collected).strip(), index
        while index < len(lines):
            collected.append(lines[index].rstrip())
            if "\\]" in lines[index]:
                index += 1
                break
            index += 1
        return "\n".join(collected).strip(), index

    while index < len(lines):
        collected.append(lines[index].rstrip())
        if _MATH_ENV_END.search(lines[index]):
            index += 1
            break
        index += 1
    return "\n".join(collected).strip(), index


def _looks_like_table_line(stripped: str) -> bool:
    return "|" in stripped and len(stripped.split("|")) >= 3


def _collect_table(lines: list[str], start: int) -> tuple[str, int]:
    collected = []
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or not _looks_like_table_line(stripped):
            break
        collected.append(lines[index].rstrip())
        index += 1
    return "\n".join(collected).strip(), index


def _join_blocks(blocks: Iterable[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip()).strip()


def _overlap_tail(blocks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []
    tail: list[str] = []
    for block in reversed(blocks):
        candidate = [block, *tail]
        if len(_join_blocks(candidate)) > chunk_overlap:
            if tail:
                break
            continue
        tail = candidate
        if len(_join_blocks(tail)) >= chunk_overlap:
            break
    return tail


def _split_oversized_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    stripped = block.strip()
    if _is_math_block(stripped):
        return _split_oversized_math_block(stripped, chunk_size, chunk_overlap)
    return _fallback_split(stripped, chunk_size, chunk_overlap)


def _is_math_block(block: str) -> bool:
    stripped = block.strip()
    return (
        stripped.startswith("$$")
        or stripped.startswith("\\[")
        or bool(_MATH_ENV_START.search(stripped))
        or stripped == "<!-- formula-not-decoded -->"
    )


def _split_oversized_math_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    stripped = block.strip()
    opener, closer, body = _math_delimiters_and_body(stripped)
    if body is None:
        return _split_lines_preserving_wrappers(stripped, "", "", chunk_size, chunk_overlap)
    return _split_lines_preserving_wrappers(body, opener, closer, chunk_size, chunk_overlap)


def _math_delimiters_and_body(block: str) -> tuple[str, str, str | None]:
    stripped = block.strip()
    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
        return "$$", "$$", stripped[2:-2].strip()
    if stripped.startswith("\\[") and stripped.endswith("\\]") and len(stripped) > 4:
        return "\\[", "\\]", stripped[2:-2].strip()
    return "", "", None


def _split_lines_preserving_wrappers(
    text: str,
    opener: str,
    closer: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    chunks: list[str] = []
    current: list[str] = []
    wrapper_size = len(opener) + len(closer) + (4 if opener and closer else 0)
    budget = max(1, chunk_size - wrapper_size)

    def emit(lines_to_emit: list[str]) -> None:
        body = "\n".join(lines_to_emit).strip()
        if not body:
            return
        if opener and closer:
            chunks.append(f"{opener}\n{body}\n{closer}")
        else:
            chunks.append(body)

    for line in lines:
        if len(line) > budget:
            if current:
                emit(current)
                current = _line_overlap_tail(current, chunk_overlap)
            pieces = _fallback_split(line, budget, min(chunk_overlap, max(0, budget // 5)))
            for piece in pieces:
                emit([piece])
            current = []
            continue
        candidate = "\n".join([*current, line]).strip()
        if current and len(candidate) > budget:
            emit(current)
            current = _line_overlap_tail(current, chunk_overlap)
        current.append(line)

    if current:
        emit(current)
    return chunks


def _line_overlap_tail(lines: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for line in reversed(lines):
        total += len(line) + 1
        if total > chunk_overlap and tail:
            break
        tail.insert(0, line)
    return tail


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, max(0, chunk_size // 2)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def _dedupe_adjacent_chunks(chunks: list[str]) -> list[str]:
    deduped: list[str] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped and (not deduped or deduped[-1] != stripped):
            deduped.append(stripped)
    return deduped


def _with_heading_context(chunks: list[str], blocks: list[str], chunk_size: int) -> list[str]:
    heading = next((block.strip() for block in blocks[:3] if block.strip().startswith("#")), "")
    if not heading:
        return chunks

    contextualized: list[str] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped.startswith("#"):
            contextualized.append(stripped)
            continue
        candidate = f"{heading}\n\n{stripped}"
        contextualized.append(candidate if len(candidate) <= chunk_size else stripped)
    return contextualized
