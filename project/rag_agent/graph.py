from agentic_rag.agents.graph import create_agent_graph as _create_agent_graph
from agentic_rag.settings import get_settings


def create_agent_graph(
    llm,
    tools_list,
    collection,
    vector_db,
    parent_store,
    workspace_memory,
    enable_reflection=True,
):
    return _create_agent_graph(
        llm=llm,
        tools_list=tools_list,
        collection=collection,
        vector_db=vector_db,
        parent_store=parent_store,
        workspace_memory=workspace_memory,
        settings=get_settings(),
        enable_reflection=enable_reflection,
    )
