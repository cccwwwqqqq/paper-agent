from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agentic_rag.settings import Settings

from .edges import route_after_orchestrator_call, route_after_rewrite
from .graph_state import AgentState, State
from .nodes import (
    aggregate_answers,
    close_reading,
    collect_answer,
    compare_papers,
    compress_context,
    fallback_response,
    finalize_interaction,
    literature_review,
    metadata_query_response,
    orchestrator,
    reflect_answer,
    request_clarification,
    rewrite_query,
    should_compress_context,
    single_paper_summary,
    summarize_history,
    verify_answer,
    workspace_memory_qa,
    workspace_inventory_response,
)


def create_agent_graph(
    llm,
    tools_list,
    collection,
    vector_db,
    parent_store,
    evidence_retriever,
    workspace_memory,
    settings: Settings,
    enable_reflection=True,
):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(
        "should_compress_context",
        partial(
            should_compress_context,
            base_token_threshold=settings.base_token_threshold,
            token_growth_factor=settings.token_growth_factor,
        )
    )
    agent_builder.add_node(collect_answer)

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator",
        partial(
            route_after_orchestrator_call,
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
        ),
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"},
    )
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)
    agent_subgraph = agent_builder.compile()

    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node(
        "rewrite_query",
        partial(
            rewrite_query,
            llm=llm,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node("request_clarification", request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node(
        "aggregate_answers",
        partial(
            aggregate_answers,
            llm=llm,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "compare_papers",
        partial(
            compare_papers,
            llm=llm,
            parent_store=parent_store,
            evidence_retriever=evidence_retriever,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "close_reading",
        partial(
            close_reading,
            llm=llm,
            evidence_retriever=evidence_retriever,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "literature_review",
        partial(
            literature_review,
            llm=llm,
            parent_store=parent_store,
            evidence_retriever=evidence_retriever,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "single_paper_summary",
        partial(
            single_paper_summary,
            llm=llm,
            parent_store=parent_store,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "workspace_memory_qa",
        partial(
            workspace_memory_qa,
            llm=llm,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "metadata_query_response",
        partial(
            metadata_query_response,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "workspace_inventory_response",
        partial(
            workspace_inventory_response,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node(
        "reflect_answer",
        partial(
            reflect_answer,
            llm=llm,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )
    graph_builder.add_node("verify_answer", partial(verify_answer, llm=llm))
    graph_builder.add_node(
        "finalize_interaction",
        partial(
            finalize_interaction,
            workspace_memory=workspace_memory,
            default_workspace_id=settings.default_workspace_id,
        ),
    )

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", "verify_answer")
    graph_builder.add_edge("close_reading", "verify_answer")
    graph_builder.add_edge("single_paper_summary", "verify_answer")
    graph_builder.add_edge("verify_answer", "finalize_interaction")
    graph_builder.add_edge("finalize_interaction", END)
    if enable_reflection:
        graph_builder.add_edge("compare_papers", "reflect_answer")
        graph_builder.add_edge("literature_review", "reflect_answer")
        graph_builder.add_edge("reflect_answer", "verify_answer")
    else:
        graph_builder.add_edge("compare_papers", "verify_answer")
        graph_builder.add_edge("literature_review", "verify_answer")
    graph_builder.add_edge("workspace_inventory_response", END)
    graph_builder.add_edge("workspace_memory_qa", END)
    graph_builder.add_edge("metadata_query_response", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])
    print("Agent graph compiled successfully.")
    return agent_graph
