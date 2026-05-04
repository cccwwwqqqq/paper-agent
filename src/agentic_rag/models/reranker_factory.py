from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from agentic_rag.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    index: int
    relevance_score: float


class SiliconFlowReranker:
    def __init__(self, model: str, api_key: str, base_url: str, timeout: float = 15.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        if not self.is_available:
            raise ValueError("SiliconFlow reranker is unavailable because the API key or base URL is missing.")

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n:
            payload["top_n"] = top_n

        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"SiliconFlow rerank HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"SiliconFlow rerank request failed: {exc.reason}") from exc

        results = body.get("results", []) or []
        parsed: list[RerankResult] = []
        for item in results:
            try:
                parsed.append(
                    RerankResult(
                        index=int(item.get("index", 0)),
                        relevance_score=float(item.get("relevance_score", 0.0) or 0.0),
                    )
                )
            except (TypeError, ValueError):
                continue
        return parsed


class NoOpReranker:
    @property
    def is_available(self) -> bool:
        return False

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        raise ValueError("No reranker backend is configured.")


def create_reranker(settings: Settings) -> Any:
    provider = settings.rerank_provider
    if provider == "siliconflow":
        if not settings.rerank_api_key:
            logger.warning("Rerank API key is missing; retrieval will fall back to heuristic reranking.")
        return SiliconFlowReranker(
            model=settings.rerank_model,
            api_key=settings.rerank_api_key,
            base_url=settings.rerank_base_url,
        )
    if provider in {"none", "disabled", ""}:
        return NoOpReranker()
    raise ValueError("Unsupported RERANK_PROVIDER '{}'. Use one of: siliconflow, none.".format(provider))
