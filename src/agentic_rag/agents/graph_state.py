from typing import Annotated, List, Set
import operator

from langgraph.graph import MessagesState


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


def concat_lists(existing: List, new: List) -> List:
    if new and new[0] == "__reset__":
        return new[1:]
    return existing + new


class State(MessagesState):
    questionIsClear: bool = False
    workspace_inventory_query: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    intent_type: str = "general_retrieval"
    task_intent: str = "single_paper_qa"
    target_papers: List[str] = []
    referenced_documents: List[str] = []
    comparison_targets: List[str] = []
    clarification_question: str = ""
    workspace_context: dict = {}
    working_memory: dict = {}
    memory_context: dict = {}
    memory_hits: List[dict] = []
    retrieval_plan: dict = {}
    answer_format: str = "short_answer"
    intent_confidence: float = 1.0
    artifact_kind: str = ""
    artifact_payload: dict = {}
    final_answer: str = ""
    verification_status: str = ""
    rerank_backend: str = ""
    paper_profiles: List[dict] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    retrieved_chunks: Annotated[List[dict], concat_lists] = []
    retrieved_parent_chunks: Annotated[List[dict], concat_lists] = []
    retrieved_paper_ids: Annotated[List[str], concat_lists] = []
    retrieved_sections: Annotated[List[str], concat_lists] = []


class AgentState(MessagesState):
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    workspace_context: dict = {}
    working_memory: dict = {}
    memory_context: dict = {}
    memory_hits: List[dict] = []
    intent_type: str = "general_retrieval"
    task_intent: str = "single_paper_qa"
    target_papers: List[str] = []
    retrieval_plan: dict = {}
    answer_format: str = "short_answer"
    intent_confidence: float = 1.0
    artifact_kind: str = ""
    artifact_payload: dict = {}
    verification_status: str = ""
    rerank_backend: str = ""
    retrieved_chunks: Annotated[List[dict], concat_lists] = []
    retrieved_parent_chunks: Annotated[List[dict], concat_lists] = []
    retrieved_paper_ids: Annotated[List[str], concat_lists] = []
    retrieved_sections: Annotated[List[str], concat_lists] = []
