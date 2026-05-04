from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from agentic_rag.settings import Settings

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[_*`#]", " ", str(text or "").lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in _normalize_text(text).split() if len(token) > 1]


class EvidenceRetriever:
    def __init__(self, settings: Settings, collection, vector_db, parent_store, reranker):
        self.settings = settings
        self.collection = collection
        self.vector_db = vector_db
        self.parent_store = parent_store
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        workspace_id: str,
        paper_id: str | None = None,
        limit: int | None = None,
        query_profile: dict[str, Any] | None = None,
        query_variants: list[str] | None = None,
    ) -> dict[str, Any]:
        profile = query_profile or {}
        variants = query_variants or [query]
        candidate_limit = max(limit or self.settings.rerank_top_n, self.settings.rerank_top_n)
        candidate_limit = min(max(candidate_limit * 4, self.settings.rerank_top_n), self.settings.rerank_max_candidates)
        qdrant_filter = self.vector_db.get_filter(workspace_id=workspace_id, paper_id=paper_id)

        deduped_hits: dict[str, tuple[Document, int]] = {}
        for query_variant in variants:
            hits = self.collection.similarity_search(
                query_variant,
                k=candidate_limit,
                score_threshold=self._candidate_threshold(profile),
                filter=qdrant_filter,
            )
            for rank, doc in enumerate(hits):
                child_id = doc.metadata.get("child_id") or f"{doc.metadata.get('parent_id', 'unknown')}::{rank}"
                if child_id not in deduped_hits:
                    deduped_hits[child_id] = (doc, rank)

        child_hits = [doc for doc, _ in deduped_hits.values()]
        parent_candidates = self._build_parent_candidates(child_hits, workspace_id=workspace_id)
        if not parent_candidates:
            return {
                "child_hits": [],
                "parent_chunks": [],
                "records": [],
                "rerank_backend": "heuristic_fallback",
            }

        top_n = min(limit or self.settings.rerank_top_n, len(parent_candidates))
        ranked_candidates, backend = self._rerank_parent_candidates(query, parent_candidates, top_n=top_n, query_profile=profile)
        records = [self._to_record(item, backend) for item in ranked_candidates[:top_n]]
        return {
            "child_hits": child_hits,
            "parent_chunks": [item["parent_chunk"] for item in ranked_candidates[:top_n]],
            "records": records,
            "rerank_backend": backend,
        }

    def _candidate_threshold(self, query_profile: dict[str, Any]) -> float:
        if query_profile.get("explicit_section_hints") or query_profile.get("is_algorithmic"):
            return 0.24
        if query_profile.get("is_performance") or query_profile.get("is_security"):
            return 0.28
        if query_profile.get("is_process"):
            return 0.30
        return 0.35

    def _build_parent_candidates(self, child_hits: list[Document], *, workspace_id: str) -> list[dict[str, Any]]:
        grouped_hits: dict[str, list[Document]] = defaultdict(list)
        for doc in child_hits:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                grouped_hits[parent_id].append(doc)

        candidates: list[dict[str, Any]] = []
        for parent_id, docs in grouped_hits.items():
            exemplar = docs[0]
            paper_id = exemplar.metadata.get("paper_id", "")
            try:
                parent_chunk = self.parent_store.load_content(workspace_id, paper_id, parent_id)
            except Exception:
                parent_chunk = {
                    "parent_id": parent_id,
                    "content": "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip()),
                    "metadata": dict(exemplar.metadata),
                }
            candidates.append(
                {
                    "parent_id": parent_id,
                    "parent_chunk": parent_chunk,
                    "metadata": parent_chunk.get("metadata", {}) or {},
                    "child_hits": docs,
                    "first_rank": min(index for _, index in [self._doc_sort_key(doc) for doc in docs]),
                }
            )
        return candidates

    @staticmethod
    def _doc_sort_key(doc: Document) -> tuple[str, int]:
        parent_id = doc.metadata.get("parent_id", "")
        child_id = doc.metadata.get("child_id", "")
        match = re.search(r"_child_(\d+)$", child_id)
        return parent_id, int(match.group(1)) if match else 0

    def _rerank_parent_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_n: int,
        query_profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        docs_for_rerank = [self._rerank_text(item["parent_chunk"]) for item in candidates]

        if getattr(self.reranker, "is_available", False):
            try:
                results = self.reranker.rerank(query, docs_for_rerank, top_n=top_n)
                if results:
                    ranked = []
                    for result in results:
                        if 0 <= result.index < len(candidates):
                            item = dict(candidates[result.index])
                            item["rank_score"] = float(result.relevance_score)
                            ranked.append(item)
                    if ranked:
                        return ranked, "siliconflow"
            except Exception as exc:
                logger.warning("SiliconFlow rerank failed; falling back to heuristic rerank: %s", exc)

        ranked = sorted(
            (dict(item, rank_score=self._heuristic_score(query, item, query_profile)) for item in candidates),
            key=lambda item: (-float(item["rank_score"]), item["first_rank"]),
        )
        return ranked[:top_n], "heuristic_fallback"

    def _heuristic_score(self, query: str, candidate: dict[str, Any], query_profile: dict[str, Any]) -> float:
        metadata = candidate.get("metadata", {}) or {}
        content = candidate.get("parent_chunk", {}).get("content", "")
        query_tokens = set(_tokenize(query))
        content_tokens = set(_tokenize(content))
        overlap = len(query_tokens & content_tokens)

        section_norm = metadata.get("section_norm") or _normalize_text(metadata.get("section", ""))
        content_type = str(metadata.get("content_type", "paragraph")).lower()
        section_bonus = 2.5 if any(hint in section_norm for hint in query_profile.get("explicit_section_hints", [])) else 0.0
        type_bonus = 0.0
        content_type_hints = {str(item).lower() for item in query_profile.get("content_type_hints", []) or []}
        effective_content_type = "formula" if content_type == "formula_refined" else content_type
        if content_type_hints and (content_type in content_type_hints or effective_content_type in content_type_hints):
            type_bonus += 2.0
        if query_profile.get("is_algorithmic") and content_type in {"algorithm", "formula", "formula_refined"}:
            type_bonus = 2.0
        elif query_profile.get("is_process") and content_type in {"algorithm", "list", "paragraph"}:
            type_bonus = 1.0
        elif query_profile.get("is_performance") and section_norm and any(
            token in section_norm for token in ("performance", "experimental", "evaluation", "results")
        ):
            type_bonus = 1.5
        if metadata.get("equation_dense") and query_profile.get("is_algorithmic"):
            type_bonus += 0.5
        if metadata.get("table_caption") and "caption" in content_type_hints:
            type_bonus += 0.5
        if metadata.get("figure_caption") and "caption" in content_type_hints:
            type_bonus += 0.5

        page_start = int(metadata.get("page_start", 9999) or 9999)
        recency_penalty = min(page_start / 1000.0, 5.0)
        return overlap + section_bonus + type_bonus - recency_penalty

    @staticmethod
    def _rerank_text(parent_chunk: dict[str, Any]) -> str:
        metadata = parent_chunk.get("metadata", {}) or {}
        return "\n".join(
            [
                f"Title: {metadata.get('title', '')}",
                f"Section: {metadata.get('section', '')}",
                f"Pages: {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}",
                f"Content Type: {metadata.get('content_type', 'paragraph')}",
                f"Equation Dense: {metadata.get('equation_dense', False)}",
                f"Table Caption: {metadata.get('table_caption', '')}",
                f"Figure Caption: {metadata.get('figure_caption', '')}",
                f"Content: {parent_chunk.get('content', '').strip()}",
            ]
        ).strip()

    @staticmethod
    def _to_record(candidate: dict[str, Any], backend: str) -> dict[str, Any]:
        parent_chunk = candidate.get("parent_chunk", {})
        metadata = parent_chunk.get("metadata", {}) or {}
        return {
            "parent_id": candidate.get("parent_id", ""),
            "paper_id": metadata.get("paper_id", ""),
            "section": metadata.get("section", ""),
            "pages": f"{metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}",
            "source": metadata.get("source", ""),
            "content": parent_chunk.get("content", "").strip(),
            "content_type": metadata.get("content_type", "paragraph"),
            "rank_score": float(candidate.get("rank_score", 0.0) or 0.0),
            "rerank_backend": backend,
        }

    @staticmethod
    def format_records(records: list[dict[str, Any]]) -> str:
        if not records:
            return "NO_RELEVANT_CHUNKS"

        return "\n\n".join(
            [
                f"Rerank Backend: {item.get('rerank_backend', 'heuristic_fallback')}\n"
                f"Rank Score: {item.get('rank_score', 0.0):.4f}\n"
                f"Parent ID: {item.get('parent_id', '')}\n"
                f"Paper ID: {item.get('paper_id', '')}\n"
                f"Section: {item.get('section', '')}\n"
                f"Pages: {item.get('pages', '')}\n"
                f"Content Type: {item.get('content_type', 'paragraph')}\n"
                f"File Name: {item.get('source', '')}\n"
                f"Content: {item.get('content', '').strip()}"
                for item in records
            ]
        )
