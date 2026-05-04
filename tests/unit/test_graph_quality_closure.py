from types import SimpleNamespace
from unittest import mock
import unittest

from agentic_rag.agents import graph as graph_module


class _FakeCompiledGraph:
    def __init__(self, builder):
        self.builder = builder


class _FakeStateGraph:
    instances = []

    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        self.compile_kwargs = {}
        _FakeStateGraph.instances.append(self)

    def add_node(self, name, node=None):
        if node is None:
            node = name
            name = getattr(node, "__name__", str(node))
        self.nodes[name] = node

    def add_edge(self, start, end):
        if isinstance(start, list):
            start = tuple(start)
        self.edges.append((start, end))

    def add_conditional_edges(self, start, route, path_map=None):
        self.conditional_edges.append((start, route, path_map))

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs
        return _FakeCompiledGraph(self)


class _FakeLlm:
    def bind_tools(self, tools):
        return self


class GraphQualityClosureTests(unittest.TestCase):
    def setUp(self):
        _FakeStateGraph.instances = []

    def test_multidoc_paths_flow_through_reflection_verification_and_finalizer(self):
        settings = SimpleNamespace(
            base_token_threshold=1000,
            token_growth_factor=0.5,
            max_iterations=3,
            max_tool_calls=6,
            default_workspace_id="demo",
        )

        with mock.patch.object(graph_module, "StateGraph", _FakeStateGraph):
            with mock.patch.object(graph_module, "ToolNode", lambda tools: "tool_node"):
                with mock.patch.object(graph_module, "InMemorySaver", lambda: "checkpointer"):
                    compiled = graph_module.create_agent_graph(
                        _FakeLlm(),
                        [],
                        collection=None,
                        vector_db=None,
                        parent_store=None,
                        evidence_retriever=None,
                        workspace_memory=None,
                        settings=settings,
                        enable_reflection=True,
                    )

        graph_builder = compiled.builder
        edges = set(graph_builder.edges)

        self.assertIn(("compare_papers", "reflect_answer"), edges)
        self.assertIn(("literature_review", "reflect_answer"), edges)
        self.assertIn(("reflect_answer", "verify_answer"), edges)
        self.assertIn(("verify_answer", "finalize_interaction"), edges)
        self.assertIn(("finalize_interaction", graph_module.END), edges)
        self.assertNotIn(("reflect_answer", graph_module.END), edges)
        self.assertNotIn(("compare_papers", graph_module.END), edges)
        self.assertNotIn(("literature_review", graph_module.END), edges)
        self.assertEqual(graph_builder.compile_kwargs["interrupt_before"], ["request_clarification"])

    def test_multidoc_no_reflection_still_flows_through_verification_and_finalizer(self):
        settings = SimpleNamespace(
            base_token_threshold=1000,
            token_growth_factor=0.5,
            max_iterations=3,
            max_tool_calls=6,
            default_workspace_id="demo",
        )

        with mock.patch.object(graph_module, "StateGraph", _FakeStateGraph):
            with mock.patch.object(graph_module, "ToolNode", lambda tools: "tool_node"):
                with mock.patch.object(graph_module, "InMemorySaver", lambda: "checkpointer"):
                    compiled = graph_module.create_agent_graph(
                        _FakeLlm(),
                        [],
                        collection=None,
                        vector_db=None,
                        parent_store=None,
                        evidence_retriever=None,
                        workspace_memory=None,
                        settings=settings,
                        enable_reflection=False,
                    )

        edges = set(compiled.builder.edges)
        self.assertIn(("compare_papers", "verify_answer"), edges)
        self.assertIn(("literature_review", "verify_answer"), edges)
        self.assertIn(("verify_answer", "finalize_interaction"), edges)
        self.assertIn(("finalize_interaction", graph_module.END), edges)


if __name__ == "__main__":
    unittest.main()
