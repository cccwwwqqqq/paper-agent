from agentic_rag.agents import nodes as _nodes
from agentic_rag.settings import get_settings

summarize_history = _nodes.summarize_history
request_clarification = _nodes.request_clarification
orchestrator = _nodes.orchestrator
fallback_response = _nodes.fallback_response
compress_context = _nodes.compress_context
collect_answer = _nodes.collect_answer

_normalize_intent_response = _nodes._normalize_intent_response
_normalize_lines = _nodes._normalize_lines
_parse_structured_tool_block = _nodes._parse_structured_tool_block
_parse_search_tool_content = _nodes._parse_search_tool_content
_parse_parent_tool_content = _nodes._parse_parent_tool_content
_record_from_doc = _nodes._record_from_doc
_record_from_parent_chunk = _nodes._record_from_parent_chunk
_unique_non_empty = _nodes._unique_non_empty
_extract_retrieval_trace_from_messages = _nodes._extract_retrieval_trace_from_messages
_heuristic_intent_analysis = _nodes._heuristic_intent_analysis
_contains_any = _nodes._contains_any
_normalize_lookup_text = _nodes._normalize_lookup_text
_extract_algorithm_terms = _nodes._extract_algorithm_terms
_extract_query_profile = _nodes._extract_query_profile
_looks_like_literature_review_query = _nodes._looks_like_literature_review_query
_query_variants = _nodes._query_variants
_section_matches_hints = _nodes._section_matches_hints
_is_process_question = _nodes._is_process_question
_is_comparison_question = _nodes._is_comparison_question
_wants_citations = _nodes._wants_citations
_section_priority = _nodes._section_priority
_detect_formula_signal = _nodes._detect_formula_signal
_fallback_close_reading_answer = _nodes._fallback_close_reading_answer
_profile_trace_from_parent_ids = _nodes._profile_trace_from_parent_ids
_dedupe_parent_chunk_records = _nodes._dedupe_parent_chunk_records
_interleave_parent_chunk_groups = _nodes._interleave_parent_chunk_groups
_profile_parent_chunk_priority = _nodes._profile_parent_chunk_priority
_select_supporting_parent_chunks = _nodes._select_supporting_parent_chunks
_support_selection_query = _nodes._support_selection_query
_response_language_instruction = _nodes._response_language_instruction
_get_or_build_paper_profile = _nodes._get_or_build_paper_profile


def rewrite_query(state, llm, workspace_memory):
    return _nodes.rewrite_query(
        state=state,
        llm=llm,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )


def should_compress_context(state):
    settings = get_settings()
    return _nodes.should_compress_context(
        state=state,
        base_token_threshold=settings.base_token_threshold,
        token_growth_factor=settings.token_growth_factor,
    )


def close_reading(state, llm, collection, vector_db, parent_store, workspace_memory):
    return _nodes.close_reading(
        state=state,
        llm=llm,
        collection=collection,
        vector_db=vector_db,
        parent_store=parent_store,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )


def _target_papers(state, workspace_memory):
    return _nodes._target_papers(
        state=state,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )


def compare_papers(state, llm, parent_store, workspace_memory):
    return _nodes.compare_papers(
        state=state,
        llm=llm,
        parent_store=parent_store,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )


def literature_review(state, llm, parent_store, workspace_memory):
    return _nodes.literature_review(
        state=state,
        llm=llm,
        parent_store=parent_store,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )


def reflect_answer(state, llm, workspace_memory):
    return _nodes.reflect_answer(
        state=state,
        llm=llm,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )


def aggregate_answers(state, llm, workspace_memory):
    return _nodes.aggregate_answers(
        state=state,
        llm=llm,
        workspace_memory=workspace_memory,
        default_workspace_id=get_settings().default_workspace_id,
    )
