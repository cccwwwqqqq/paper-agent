from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from agentic_rag.settings import Settings
from agentic_rag.utils.files import clear_directory_contents


class ParentStoreManager:
    __store_path: Path

    def __init__(self, settings: Settings, store_path: str | Path | None = None):
        self.settings = settings
        self.__store_path = Path(store_path) if store_path else Path(settings.parent_store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)

    def _workspace_dir(self, workspace_id: str | None, paper_id: str | None = None) -> Path:
        base = self.__store_path if workspace_id is None else self.__store_path / workspace_id
        if paper_id:
            base = base / paper_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def save(self, workspace_id: str, paper_id: str, parent_id: str, content: str, metadata: Dict) -> None:
        file_path = self._workspace_dir(workspace_id, paper_id) / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_many(self, workspace_id: str, paper_id: str, parents: List) -> None:
        for parent_id, doc in parents:
            self.save(workspace_id, paper_id, parent_id, doc.page_content, doc.metadata)

    def load(self, workspace_id: str, paper_id: str, parent_id: str) -> Dict:
        file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        file_path = self._workspace_dir(workspace_id, paper_id) / file_name
        return json.loads(file_path.read_text(encoding="utf-8"))

    def load_content(self, workspace_id: str, paper_id: str, parent_id: str) -> Dict:
        data = self.load(workspace_id, paper_id, parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"],
        }

    @staticmethod
    def _get_sort_key(id_str):
        match = re.search(r"_parent_(\d+)$", id_str)
        return int(match.group(1)) if match else 0

    def load_content_many(self, workspace_id: str, paper_id: str, parent_ids: List[str]) -> List[Dict]:
        unique_ids = set(parent_ids)
        return [self.load_content(workspace_id, paper_id, pid) for pid in sorted(unique_ids, key=self._get_sort_key)]

    def load_paper(self, workspace_id: str, paper_id: str) -> List[Dict]:
        paper_dir = self._workspace_dir(workspace_id, paper_id)
        parent_ids = [path.stem for path in paper_dir.glob("*.json")]
        return self.load_content_many(workspace_id, paper_id, parent_ids)

    def clear_store(self, workspace_id: str | None = None) -> None:
        target = self._workspace_dir(workspace_id) if workspace_id else self.__store_path
        target.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(target)

