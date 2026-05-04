from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.settings import Settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(text: Any) -> str:
    normalized = re.sub(r"[_*`#]", " ", str(text or "").lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: Any) -> set[str]:
    return {token for token in _normalize_text(text).split() if len(token) > 1}


def _score_match(query: str, payload: Any) -> int:
    query_tokens = _tokenize(query)
    payload_tokens = _tokenize(payload)
    return len(query_tokens & payload_tokens)


class WorkspaceMemoryStore:
    def __init__(self, settings: Settings, root_path: str | Path | None = None):
        self.settings = settings
        self.root_path = Path(root_path) if root_path else Path(settings.workspace_memory_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def ensure_workspace(self, workspace_id: str) -> Path:
        workspace_dir = self.root_path / workspace_id
        (workspace_dir / "paper_profiles").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "history").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "semantic").mkdir(parents=True, exist_ok=True)
        return workspace_dir

    def get_workspace_dir(self, workspace_id: str) -> Path:
        return self.root_path / workspace_id

    def register_paper(self, workspace_id: str, paper_record: dict[str, Any]) -> None:
        workspace_dir = self.ensure_workspace(workspace_id)
        catalog_path = workspace_dir / "catalog.json"
        catalog = self._read_json(catalog_path, default={"papers": []})

        papers = [paper for paper in catalog["papers"] if paper.get("paper_id") != paper_record["paper_id"]]
        papers.append({**paper_record, "updated_at": _utc_now()})
        catalog["papers"] = sorted(papers, key=lambda item: item.get("paper_id", ""))
        self._write_json(catalog_path, catalog)

    def list_papers(self, workspace_id: str) -> list[dict[str, Any]]:
        workspace_dir = self.get_workspace_dir(workspace_id)
        if not workspace_dir.exists():
            return []
        catalog = self._read_json(workspace_dir / "catalog.json", default={"papers": []})
        return catalog.get("papers", [])

    def save_paper_profile(self, workspace_id: str, paper_id: str, profile: dict[str, Any]) -> None:
        workspace_dir = self.ensure_workspace(workspace_id)
        payload = {"paper_id": paper_id, "updated_at": _utc_now(), **profile}
        self._write_json(workspace_dir / "paper_profiles" / f"{paper_id}.json", payload)
        self.save_semantic_fact(
            workspace_id,
            f"paper_profile::{paper_id}",
            {
                "kind": "paper_profile",
                "paper_id": paper_id,
                "title": payload.get("title", ""),
                "core_method": payload.get("core_method", ""),
                "source_sections": payload.get("source_sections", []),
            },
        )

    def load_paper_profile(self, workspace_id: str, paper_id: str) -> dict[str, Any] | None:
        workspace_dir = self.get_workspace_dir(workspace_id)
        if not workspace_dir.exists():
            return None
        path = workspace_dir / "paper_profiles" / f"{paper_id}.json"
        if not path.exists():
            return None
        return self._read_json(path, default=None)

    def list_paper_profiles(self, workspace_id: str, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        profiles = []
        for paper in self.list_papers(workspace_id):
            paper_id = paper["paper_id"]
            if paper_ids and paper_id not in paper_ids:
                continue
            profile = self.load_paper_profile(workspace_id, paper_id)
            if profile:
                profiles.append(profile)
        return profiles

    def save_working_memory_snapshot(self, workspace_id: str, snapshot: dict[str, Any]) -> None:
        workspace_dir = self.ensure_workspace(workspace_id)
        self._write_json(workspace_dir / "working_memory.json", {"updated_at": _utc_now(), **snapshot})

    def load_working_memory_snapshot(self, workspace_id: str) -> dict[str, Any]:
        workspace_dir = self.get_workspace_dir(workspace_id)
        if not workspace_dir.exists():
            return {}
        return self._read_json(workspace_dir / "working_memory.json", default={})

    def search_paper_profiles(self, workspace_id: str, query: str, limit: int = 3) -> list[dict[str, Any]]:
        ranked = []
        for profile in self.list_paper_profiles(workspace_id):
            score = _score_match(query, json.dumps(profile, ensure_ascii=False))
            if score > 0:
                ranked.append({**profile, "_score": score})
        ranked.sort(key=lambda item: (-item["_score"], item.get("paper_id", "")))
        return ranked[:limit]

    def search_interactions(self, workspace_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.search_episodic_memory(workspace_id, query, limit=limit)

    def search_episodic_memory(self, workspace_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        workspace_dir = self.get_workspace_dir(workspace_id)
        history_path = workspace_dir / "history" / "interactions.json"
        history = self._read_json(history_path, default={"items": []})
        ranked = []
        for item in history.get("items", []):
            score = _score_match(query, json.dumps(item, ensure_ascii=False))
            if score > 0:
                ranked.append({**item, "_score": score})
        ranked.sort(key=lambda item: (-item["_score"], item.get("created_at", "")), reverse=False)
        return ranked[:limit]

    def search_working_memory(self, workspace_id: str, query: str) -> dict[str, Any]:
        snapshot = self.load_working_memory_snapshot(workspace_id)
        if not snapshot:
            return {}

        if not query.strip():
            return {key: value for key, value in snapshot.items() if value not in ("", [], {}, None)}

        query_tokens = _tokenize(query)
        filtered = {}
        for key, value in snapshot.items():
            if not value or key == "updated_at":
                continue
            value_tokens = _tokenize(value)
            if query_tokens & value_tokens or key in {
                "focus_paper_id",
                "current_dialogue_paper_id",
                "current_research_question",
                "recent_papers",
                "term_aliases",
            }:
                filtered[key] = value
        return filtered

    def search_working_memory_facts(self, workspace_id: str, query: str) -> dict[str, Any]:
        return self.search_working_memory(workspace_id, query)

    def load_semantic_memory(self, workspace_id: str) -> dict[str, Any]:
        workspace_dir = self.get_workspace_dir(workspace_id)
        semantic_dir = workspace_dir / "semantic"
        if not semantic_dir.exists():
            return {}
        items = {}
        for path in sorted(semantic_dir.glob("*.json")):
            items[path.stem] = self._read_json(path, default={})
        return items

    def save_semantic_fact(self, workspace_id: str, fact_id: str, payload: dict[str, Any]) -> None:
        workspace_dir = self.ensure_workspace(workspace_id)
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(fact_id or "").strip()) or "fact"
        self._write_json(
            workspace_dir / "semantic" / f"{safe_name}.json",
            {"fact_id": fact_id, "updated_at": _utc_now(), **payload},
        )

    def search_semantic_memory(self, workspace_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        ranked = []
        for fact in self.load_semantic_memory(workspace_id).values():
            score = _score_match(query, json.dumps(fact, ensure_ascii=False))
            if score > 0:
                ranked.append({**fact, "_score": score})
        ranked.sort(key=lambda item: (-item["_score"], item.get("fact_id", "")))
        return ranked[:limit]

    def record_interaction(self, workspace_id: str, entry: dict[str, Any]) -> None:
        workspace_dir = self.ensure_workspace(workspace_id)
        history_path = workspace_dir / "history" / "interactions.json"
        history = self._read_json(history_path, default={"items": []})
        items = history.get("items", [])
        items.append(
            {
                "created_at": _utc_now(),
                "query": entry.get("query", ""),
                "intent_type": entry.get("intent_type", ""),
                "paper_ids": entry.get("paper_ids", []),
                "retrieved_sections": entry.get("retrieved_sections", []),
                "verification_status": entry.get("verification_status", ""),
                "artifact_kind": entry.get("artifact_kind", ""),
                **entry,
            }
        )
        history["items"] = items[-100:]
        self._write_json(history_path, history)

    def clear_workspace(self, workspace_id: str) -> None:
        workspace_dir = self.ensure_workspace(workspace_id)
        shutil.rmtree(workspace_dir, ignore_errors=True)
        self.ensure_workspace(workspace_id)

    @staticmethod
    def _read_json(path: Path, default: Any):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
