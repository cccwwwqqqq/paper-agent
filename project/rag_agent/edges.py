from agentic_rag.agents.edges import route_after_rewrite
from agentic_rag.agents.edges import route_after_orchestrator_call as _route_after_orchestrator_call
from agentic_rag.settings import get_settings


def route_after_orchestrator_call(state):
    settings = get_settings()
    return _route_after_orchestrator_call(
        state,
        max_iterations=settings.max_iterations,
        max_tool_calls=settings.max_tool_calls,
    )


__all__ = ["route_after_orchestrator_call", "route_after_rewrite"]
