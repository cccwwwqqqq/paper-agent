from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class WorkspaceContext(BaseModel):
    workspace_id: str = Field(description="Current workspace identifier.")
    focus_paper_id: Optional[str] = Field(default=None, description="Focused paper for close reading.")
    knowledge_scope: Literal["workspace_documents", "workspace_memory", "global_knowledge"] = Field(
        default="workspace_documents",
        description="Knowledge scope allowed for the answer.",
    )


class WorkingMemory(BaseModel):
    workspace_id: str = ""
    focus_paper_id: Optional[str] = None
    current_dialogue_paper_id: Optional[str] = None
    current_research_question: str = ""
    term_aliases: dict[str, str] = Field(default_factory=dict)
    recent_papers: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recent_summary: str = ""


class IntentAnalysis(BaseModel):
    intent_type: Literal[
        "general_retrieval",
        "single_doc_close_reading",
        "cross_doc_comparison",
        "literature_review",
    ] = Field(description="Detected literature-reading intent.")
    task_intent: Literal[
        "single_paper_qa",
        "single_paper_summary",
        "method_explanation",
        "multi_paper_comparison",
        "literature_review",
        "citation_finding",
        "workspace_memory_qa",
        "metadata_query",
    ] = Field(default="single_paper_qa", description="Fine-grained task intent used to tune retrieval and generation.")
    resolved_query: str = Field(description="Resolved, self-contained absolute query.")
    rewritten_questions: list[str] = Field(default_factory=list, description="Sub-queries for retrieval.")
    referenced_documents: list[str] = Field(default_factory=list, description="Referenced paper ids if any.")
    comparison_targets: list[str] = Field(default_factory=list, description="Target papers for comparison/review.")
    target_papers: list[str] = Field(default_factory=list, description="Target paper ids resolved for the fine-grained task.")
    topic: str = Field(default="", description="Main research topic or task focus.")
    question_type: str = Field(default="", description="Question operation such as answer, summarize, explain, compare, find_evidence, or query_metadata.")
    retrieval_scope: list[str] = Field(default_factory=list, description="Preferred paper sections or evidence scopes for retrieval.")
    answer_format: str = Field(default="short_answer", description="Preferred response format.")
    need_memory: bool = Field(default=False, description="Whether workspace memory should be prioritized.")
    need_metadata_filter: bool = Field(default=False, description="Whether structured metadata should be prioritized over semantic retrieval.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Intent classification confidence.")
    retrieval_plan: dict = Field(default_factory=dict, description="Minimal retrieval plan for the downstream retriever.")
    artifact_kind: str = Field(default="", description="Optional inline artifact to build from the answer.")
    needs_clarification: bool = Field(description="Whether user clarification is required.")
    clarification_question: str = Field(default="", description="Clarification question for the user.")


class PaperProfile(BaseModel):
    paper_id: str
    title: str = ""
    problem: str = ""
    core_method: str = ""
    assumptions: list[str] = Field(default_factory=list)
    dataset_or_setting: str = ""
    metrics: list[str] = Field(default_factory=list)
    main_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)


class AnswerVerification(BaseModel):
    verification_status: Literal["pass", "downgrade"] = Field(description="Whether the answer is sufficiently grounded.")
    revised_answer: str = Field(description="Final answer after evidence-grounding revisions.")
