import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessageChunk, ToolMessage

from agentic_rag.services.chat_service import ChatService


def _find_message(messages, node):
    for message in messages:
        metadata = message.get("metadata", {}) or {}
        if metadata.get("node") == node:
            return message
    return None


def _snapshot_event(event):
    if not isinstance(event, list):
        return event
    snapshot = []
    for message in event:
        copied = dict(message)
        if "metadata" in copied and copied["metadata"] is not None:
            copied["metadata"] = dict(copied["metadata"])
        snapshot.append(copied)
    return snapshot


class FakeGraph:
    def __init__(self):
        self.next = False
        self.update_calls = []
        self.stream_inputs = []
        self.values = {}

    def get_state(self, config):
        return SimpleNamespace(next=self.next, values=self.values)

    def update_state(self, config, state_update):
        self.update_calls.append((config, state_update))

    def stream(self, stream_input, config=None, stream_mode=None):
        self.stream_inputs.append((stream_input, config, stream_mode))
        yield AIMessageChunk(
            content='{"intent_type":"general_retrieval","resolved_query":"What changed?","referenced_documents":["paper-1"]}'
        ), {"langgraph_node": "rewrite_query"}
        yield SimpleNamespace(tool_calls=[{"name": "search_child_chunks", "id": "tool-1"}]), {
            "langgraph_node": "orchestrator"
        }
        yield ToolMessage(content="NO_RELEVANT_CHUNKS", tool_call_id="tool-1"), {"langgraph_node": "tools"}
        yield AIMessageChunk(content="Final supported answer."), {"langgraph_node": "reflect_answer"}


class FakeWorkspaceMemory:
    def __init__(self):
        self.loaded_workspaces = []

    def load_working_memory_snapshot(self, workspace_id):
        self.loaded_workspaces.append(workspace_id)
        return {"workspace_id": workspace_id}


class FakeObservability:
    def __init__(self):
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1


class FakeRagSystem:
    def __init__(self):
        self.agent_graph = FakeGraph()
        self.workspace_memory = FakeWorkspaceMemory()
        self.observability = FakeObservability()
        self.reset_calls = 0
        self.active_context = {
            "workspace_context": {
                "workspace_id": "default",
                "focus_paper_id": None,
                "selected_focus_paper_id": None,
                "intent_type": "general_retrieval",
            }
        }

    def set_workspace_context(self, workspace_id, focus_paper_id=None, intent_type="general_retrieval"):
        previous_context = self.active_context["workspace_context"]
        if focus_paper_id != previous_context.get("selected_focus_paper_id"):
            resolved_focus = focus_paper_id
        else:
            resolved_focus = previous_context.get("focus_paper_id")
        self.active_context = {
            "workspace_context": {
                "workspace_id": workspace_id,
                "focus_paper_id": resolved_focus,
                "selected_focus_paper_id": focus_paper_id,
                "intent_type": intent_type,
            }
        }

    def sync_workspace_context(self, workspace_context):
        if not workspace_context:
            return
        current = self.active_context["workspace_context"]
        self.active_context = {
            "workspace_context": {
                "workspace_id": workspace_context.get("workspace_id", current.get("workspace_id")),
                "focus_paper_id": workspace_context.get("focus_paper_id", current.get("focus_paper_id")),
                "selected_focus_paper_id": workspace_context.get(
                    "selected_focus_paper_id",
                    current.get("selected_focus_paper_id"),
                ),
                "intent_type": workspace_context.get("intent_type", current.get("intent_type")),
            }
        }

    def get_workspace_context(self):
        return self.active_context

    def get_config(self):
        return {"configurable": {"thread_id": "fake-thread"}}

    def reset_thread(self):
        self.reset_calls += 1


class ChatFlowIntegrationTests(unittest.TestCase):
    def test_chat_service_streams_query_planning_tool_preview_and_final_answer(self):
        rag_system = FakeRagSystem()
        service = ChatService(rag_system)

        events = [_snapshot_event(event) for event in service.chat("What changed?", [], "workspace-a", focus_paper_id="paper-1")]

        self.assertGreaterEqual(len(events), 2)
        first_event = events[0]
        last_event = events[-1]

        self.assertEqual(rag_system.active_context["workspace_context"]["workspace_id"], "workspace-a")
        self.assertEqual(rag_system.active_context["workspace_context"]["focus_paper_id"], "paper-1")
        self.assertEqual(rag_system.workspace_memory.loaded_workspaces, ["workspace-a"])
        self.assertEqual(rag_system.agent_graph.stream_inputs[0][2], "messages")
        self.assertEqual(rag_system.agent_graph.stream_inputs[0][0]["workspace_context"]["workspace_id"], "workspace-a")

        status_message = _find_message(first_event, "status")
        self.assertIsNotNone(status_message)
        self.assertTrue(
            any(token in status_message["content"] for token in ["Analyzing the query", "正在分析问题"]),
            status_message["content"],
        )

        rewrite_message = _find_message(last_event, "rewrite_query")
        self.assertIsNotNone(rewrite_message)
        self.assertTrue(
            any(token in rewrite_message["content"] for token in ["Intent: `general_retrieval`", "意图类型：`general_retrieval`"]),
            rewrite_message["content"],
        )

        final_message = _find_message(last_event, "reflect_answer")
        self.assertIsNotNone(final_message)
        self.assertIn(final_message["metadata"]["title"], {"Final Answer", "最终答案"})
        self.assertIn("Final supported answer.", final_message["content"])

        tool_message = next(
            message
            for message in last_event
            if message.get("metadata", {}).get("title") in {"Tool: Search child chunks", "工具：检索相关片段"}
        )
        self.assertTrue(
            any(
                token in tool_message["content"]
                for token in ["No relevant chunks found.", "未检索到相关片段", "未找到相关片段。"]
            ),
            tool_message["content"],
        )

    def test_chat_service_resumes_interrupted_thread_with_update_state(self):
        rag_system = FakeRagSystem()
        rag_system.agent_graph.next = True
        service = ChatService(rag_system)

        list(service.chat("Resume this thread", [], "workspace-b"))

        self.assertEqual(len(rag_system.agent_graph.update_calls), 1)
        update_payload = rag_system.agent_graph.update_calls[0][1]
        self.assertEqual(update_payload["workspace_context"]["workspace_id"], "workspace-b")
        self.assertIsNone(rag_system.agent_graph.stream_inputs[0][0])

    def test_chat_service_surfaces_clarification_after_interrupt(self):
        rag_system = FakeRagSystem()
        rag_system.agent_graph.next = ("request_clarification",)
        rag_system.agent_graph.values = {"clarification_question": "Which paper do you mean by BE-PBAC?"}
        service = ChatService(rag_system)

        events = [_snapshot_event(event) for event in service.chat("BE-PBAC这篇论文讲了什么", [], "workspace-c")]

        last_event = events[-1]
        clarification_message = _find_message(last_event, "clarification")
        self.assertIsNotNone(clarification_message)
        self.assertIn("BE-PBAC", clarification_message["content"])

    def test_clear_session_resets_thread_and_flushes_observability(self):
        rag_system = FakeRagSystem()
        service = ChatService(rag_system)

        service.clear_session()

        self.assertEqual(rag_system.reset_calls, 1)
        self.assertEqual(rag_system.observability.flush_calls, 1)

    def test_chat_service_preserves_resolved_focus_when_dropdown_selection_is_unchanged(self):
        rag_system = FakeRagSystem()
        rag_system.active_context = {
            "workspace_context": {
                "workspace_id": "paper-test",
                "focus_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                "selected_focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                "intent_type": "single_doc_close_reading",
            }
        }
        rag_system.agent_graph.values = {
            "workspace_context": {
                "workspace_id": "paper-test",
                "focus_paper_id": "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
                "selected_focus_paper_id": "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
                "intent_type": "single_doc_close_reading",
            }
        }
        service = ChatService(rag_system)

        list(service.chat("What is the biggest difference between it and PM-ABE?", [], "paper-test", focus_paper_id="pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa"))

        stream_input = rag_system.agent_graph.stream_inputs[0][0]
        self.assertEqual(
            stream_input["workspace_context"]["focus_paper_id"],
            "a-cloud-edge-device-collaborative-attribute-base-499d8e10",
        )
        self.assertEqual(
            stream_input["workspace_context"]["selected_focus_paper_id"],
            "pm-abe-puncturable-bilateral-fine-grained-access-7afb2cfa",
        )


if __name__ == "__main__":
    unittest.main()
