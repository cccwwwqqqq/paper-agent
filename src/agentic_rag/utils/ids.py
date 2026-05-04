from __future__ import annotations

import hashlib
import re
from pathlib import Path

SAFE_PAPER_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{8}$")


def make_safe_paper_id(raw_name: str, max_slug_length: int = 48) -> str:
    stem = Path(raw_name).stem if raw_name else "paper"
    normalized = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    normalized = normalized[:max_slug_length].strip("-") or "paper"
    suffix = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{suffix}"


def resolve_paper_id(raw_name: str, max_slug_length: int = 48) -> str:
    stem = Path(raw_name).stem if raw_name else "paper"
    if SAFE_PAPER_ID_PATTERN.fullmatch(stem):
        return stem
    return make_safe_paper_id(stem, max_slug_length=max_slug_length)

