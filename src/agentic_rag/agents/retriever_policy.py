from typing import Optional

from pydantic import BaseModel, Field


class RetrieverPolicy(BaseModel):
    workspace_id: str
    intent_type: str = "general_retrieval"
    task_intent: str = "single_paper_qa"
    focus_paper_id: Optional[str] = None
    knowledge_scope: str = "workspace_documents"
    mode: str = "general"
    paper_ids: list[str] = Field(default_factory=list)
    section_hints: list[str] = Field(default_factory=list)
    content_type_hints: list[str] = Field(default_factory=list)
    retrieval_scope: list[str] = Field(default_factory=list)
    answer_format: str = "short_answer"
    need_memory: bool = False
    need_metadata_filter: bool = False
    per_paper_limit: int = 3
    global_limit: int = 6

    def effective_paper_id(self, requested_paper_id: Optional[str] = None) -> Optional[str]:
        return self.focus_paper_id or requested_paper_id

    def effective_paper_ids(self) -> list[str]:
        if self.paper_ids:
            return [paper_id for paper_id in self.paper_ids if paper_id]
        if self.focus_paper_id:
            return [self.focus_paper_id]
        return []

    @classmethod
    def from_context(cls, workspace_context: dict, default_workspace_id: str, default_knowledge_scope: str):
        retrieval_plan = workspace_context.get("retrieval_plan", {}) or {}
        return cls(
            workspace_id=workspace_context.get("workspace_id", default_workspace_id),
            intent_type=workspace_context.get("intent_type", "general_retrieval"),
            task_intent=workspace_context.get("task_intent", retrieval_plan.get("task_intent", "single_paper_qa")),
            focus_paper_id=workspace_context.get("focus_paper_id"),
            knowledge_scope=workspace_context.get("knowledge_scope", default_knowledge_scope),
            mode=retrieval_plan.get("mode", "general"),
            paper_ids=retrieval_plan.get("paper_ids", []) or [],
            section_hints=retrieval_plan.get("section_hints", []) or [],
            content_type_hints=retrieval_plan.get("content_type_hints", []) or [],
            retrieval_scope=retrieval_plan.get("retrieval_scope", []) or [],
            answer_format=retrieval_plan.get("answer_format", workspace_context.get("answer_format", "short_answer")),
            need_memory=bool(retrieval_plan.get("need_memory", False)),
            need_metadata_filter=bool(retrieval_plan.get("need_metadata_filter", False)),
            per_paper_limit=int(retrieval_plan.get("per_paper_limit", 3) or 3),
            global_limit=int(retrieval_plan.get("global_limit", 6) or 6),
        )
