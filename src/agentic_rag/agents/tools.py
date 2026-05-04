from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from agentic_rag.agents.retriever_policy import RetrieverPolicy
from agentic_rag.settings import Settings


class ToolFactory:
    def __init__(self, settings: Settings, evidence_retriever, parent_store_manager, workspace_memory, context_provider):
        self.settings = settings
        self.evidence_retriever = evidence_retriever
        self.parent_store_manager = parent_store_manager
        self.workspace_memory = workspace_memory
        self.context_provider = context_provider

    def _get_policy(self) -> RetrieverPolicy:
        raw_context = self.context_provider() or {}
        workspace_context = raw_context.get("workspace_context", raw_context)
        return RetrieverPolicy.from_context(
            workspace_context,
            default_workspace_id=self.settings.default_workspace_id,
            default_knowledge_scope=self.settings.knowledge_scope,
        )

    def _search_child_chunks(self, query: str, limit: int | None = None, paper_id: str = "") -> str:
        """Search child chunks inside the active workspace, optionally constrained to one paper."""
        try:
            policy = self._get_policy()
            effective_paper_id = policy.effective_paper_id(paper_id or None)
            result = self.evidence_retriever.search(
                query,
                workspace_id=policy.workspace_id,
                paper_id=effective_paper_id,
                limit=limit or self.settings.default_search_limit,
            )
            if not result["records"]:
                return "NO_RELEVANT_CHUNKS"
            return self.evidence_retriever.format_records(result["records"])
        except Exception as exc:
            return f"RETRIEVAL_ERROR: {exc}"

    def _retrieve_parent_chunks(self, parent_id: str, paper_id: str = "") -> str:
        """Retrieve a parent chunk by id from the active workspace and paper context."""
        try:
            policy = self._get_policy()
            inferred_paper_id = parent_id.rsplit("_parent_", 1)[0] if "_parent_" in parent_id else None
            effective_paper_id = policy.effective_paper_id(paper_id or inferred_paper_id)
            if not effective_paper_id:
                return "PARENT_RETRIEVAL_ERROR: paper_id is required for parent retrieval."

            parent = self.parent_store_manager.load_content(policy.workspace_id, effective_paper_id, parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"Paper ID: {parent.get('metadata', {}).get('paper_id', effective_paper_id)}\n"
                f"Section: {parent.get('metadata', {}).get('section', 'unknown')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
        except Exception as exc:
            return f"PARENT_RETRIEVAL_ERROR: {exc}"

    def _list_workspace_papers(self) -> str:
        """List the papers currently registered in the active workspace."""
        policy = self._get_policy()
        papers = self.workspace_memory.list_papers(policy.workspace_id)
        if not papers:
            return "NO_WORKSPACE_PAPERS"
        return "\n".join(
            f"- {paper.get('paper_id')} ({paper.get('source_name', paper.get('title', 'unknown'))})"
            for paper in papers
        )

    def create_tools(self) -> List:
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        list_papers_tool = tool("list_workspace_papers")(self._list_workspace_papers)
        return [search_tool, retrieve_tool, list_papers_tool]
