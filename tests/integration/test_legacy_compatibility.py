import os
import unittest
from unittest import mock

import project.core.chat_interface as legacy_chat_interface
import project.core.document_manager as legacy_document_manager
import project.core.embedding_factory as legacy_embedding_factory
import project.core.llm_factory as legacy_llm_factory
import project.core.rag_system as legacy_rag_system
import project.rag_agent.edges as legacy_edges
import project.rag_agent.graph as legacy_agent_graph
import project.rag_agent.nodes as legacy_agent_nodes
import project.rag_agent.tools as legacy_agent_tools
import project.parsers as legacy_parsers
import project.ui.gradio_app as legacy_gradio_app
import project.utils as legacy_utils
from agentic_rag.parsers.adapters import PymuPdf4LlmParserAdapter
from agentic_rag.services.chat_service import ChatInterface as NewChatInterface
from agentic_rag.services.ingestion_service import IngestionService
from agentic_rag.settings import load_settings, reset_settings_cache


class LegacyCompatibilityTests(unittest.TestCase):
    def test_chat_interface_reexports_new_service(self):
        self.assertIs(legacy_chat_interface.ChatInterface, NewChatInterface)

    def test_document_manager_reexports_ingestion_service(self):
        self.assertTrue(issubclass(legacy_document_manager.DocumentManager, IngestionService))

    def test_legacy_rag_system_is_lazy_until_initialize(self):
        rag_system = legacy_rag_system.RAGSystem()

        self.assertIsNone(rag_system._delegate)
        with self.assertRaisesRegex(AttributeError, "initialize"):
            _ = rag_system.settings

    def test_legacy_parser_factory_uses_new_parser(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            clean_settings = load_settings(use_env=False)
            with mock.patch.object(legacy_parsers, "get_settings", return_value=clean_settings):
                reset_settings_cache()
                parser = legacy_parsers.create_pdf_parser()
                reset_settings_cache()

        self.assertIsInstance(parser, PymuPdf4LlmParserAdapter)

    def test_legacy_utils_reexport_id_helpers(self):
        paper_id = legacy_utils.resolve_paper_id("My Paper.pdf")

        self.assertRegex(paper_id, r"^my-paper-[0-9a-f]{8}$")

    def test_legacy_llm_factory_delegates_to_new_factory(self):
        with mock.patch.object(legacy_llm_factory, "_create_llm", return_value="llm") as create_mock:
            result = legacy_llm_factory.create_llm()

        self.assertEqual(result, "llm")
        create_mock.assert_called_once()

    def test_legacy_embedding_factory_delegates_to_new_factory(self):
        with mock.patch.object(
            legacy_embedding_factory,
            "_create_dense_embeddings",
            return_value="embeddings",
        ) as create_mock:
            result = legacy_embedding_factory.create_dense_embeddings()

        self.assertEqual(result, "embeddings")
        create_mock.assert_called_once()

    def test_legacy_gradio_wrapper_builds_runtime_then_delegates(self):
        runtime = object()
        demo = object()

        with mock.patch.object(legacy_gradio_app, "build_runtime", return_value=runtime) as build_mock:
            with mock.patch.object(legacy_gradio_app, "_create_gradio_ui", return_value=demo) as ui_mock:
                result = legacy_gradio_app.create_gradio_ui()

        self.assertIs(result, demo)
        build_mock.assert_called_once_with()
        ui_mock.assert_called_once_with(runtime)

    def test_legacy_agent_graph_injects_settings(self):
        graph = object()
        settings = load_settings(use_env=False)

        with mock.patch.object(legacy_agent_graph, "get_settings", return_value=settings):
            with mock.patch.object(legacy_agent_graph, "_create_agent_graph", return_value=graph) as create_mock:
                result = legacy_agent_graph.create_agent_graph(
                    llm="llm",
                    tools_list=["tool"],
                    collection="collection",
                    vector_db="vector_db",
                    parent_store="parent_store",
                    workspace_memory="workspace_memory",
                    enable_reflection=False,
                )

        self.assertIs(result, graph)
        create_mock.assert_called_once()
        self.assertEqual(create_mock.call_args.kwargs["settings"], settings)
        self.assertFalse(create_mock.call_args.kwargs["enable_reflection"])

    def test_legacy_tool_factory_injects_settings(self):
        settings = load_settings(use_env=False)

        with mock.patch.object(legacy_agent_tools, "get_settings", return_value=settings):
            with mock.patch.object(legacy_agent_tools._ToolFactory, "__init__", return_value=None) as init_mock:
                legacy_agent_tools.ToolFactory(
                    collection="collection",
                    vector_db="vector_db",
                    parent_store_manager="parent_store",
                    workspace_memory="workspace_memory",
                    context_provider=lambda: {},
                )

        init_mock.assert_called_once()
        self.assertEqual(init_mock.call_args.kwargs["settings"], settings)

    def test_legacy_edges_inject_runtime_limits(self):
        settings = load_settings(use_env=False)

        with mock.patch.object(legacy_edges, "get_settings", return_value=settings):
            with mock.patch.object(legacy_edges, "_route_after_orchestrator_call", return_value="tools") as route_mock:
                result = legacy_edges.route_after_orchestrator_call({"messages": []})

        self.assertEqual(result, "tools")
        route_mock.assert_called_once_with(
            {"messages": []},
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
        )

    def test_legacy_nodes_rewrite_query_injects_default_workspace(self):
        settings = load_settings(use_env=False)

        with mock.patch.object(legacy_agent_nodes, "get_settings", return_value=settings):
            with mock.patch.object(legacy_agent_nodes._nodes, "rewrite_query", return_value={"ok": True}) as rewrite_mock:
                result = legacy_agent_nodes.rewrite_query("state", "llm", "workspace_memory")

        self.assertEqual(result, {"ok": True})
        rewrite_mock.assert_called_once_with(
            state="state",
            llm="llm",
            workspace_memory="workspace_memory",
            default_workspace_id=settings.default_workspace_id,
        )


if __name__ == "__main__":
    unittest.main()
