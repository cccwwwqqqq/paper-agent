from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

SILENT_NODES = {"rewrite_query"}
SYSTEM_NODES = {"summarize_history", "rewrite_query"}

SYSTEM_NODE_CONFIG = {
    "rewrite_query": {"title": "问题规划"},
    "summarize_history": {"title": "对话摘要"},
}

TOOL_DISPLAY_NAMES = {
    "search_child_chunks": "检索相关片段",
    "retrieve_parent_chunks": "提取原文段落",
    "list_workspace_papers": "列出工作区论文",
}
VISIBLE_TOOL_NAMES = set(TOOL_DISPLAY_NAMES)


def make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    return next((i for i, item in enumerate(messages) if item.get("metadata", {}).get("node") == node), None)


def parse_rewrite_json(buffer):
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_tool_preview(content):
    preview = str(content)
    replacements = {
        "NO_RELEVANT_CHUNKS": "未找到相关片段。",
        "NO_PARENT_DOCUMENT": "未找到对应的上级文档。",
        "NO_WORKSPACE_PAPERS": "当前工作区中暂无论文。",
        "PARENT_RETRIEVAL_ERROR:": "上级文档检索错误：",
        "RETRIEVAL_ERROR:": "检索错误：",
        "Parent ID:": "父级片段 ID：",
        "Paper ID:": "论文 ID：",
        "Section:": "章节：",
        "Pages:": "页码：",
        "File Name:": "文件名：",
        "Source:": "来源：",
        "Content:": "内容：",
    }
    for source, target in replacements.items():
        preview = preview.replace(source, target)
    return preview


def format_rewrite_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "正在分析问题并准备检索..."

    lines = [f"意图类型：`{data.get('intent_type', 'unknown')}`"]
    if data.get("task_intent"):
        lines.append(f"\nTask intent: `{data.get('task_intent')}`")
    if data.get("resolved_query"):
        lines.append(f"\n解析后的问题：{data['resolved_query']}")
    if data.get("referenced_documents"):
        lines.append("\n识别到的相关文档：")
        lines.extend(f"- {paper}" for paper in data["referenced_documents"])
    if data.get("needs_clarification"):
        lines.append(f"\n需要补充说明：{data.get('clarification_question', '')}")
    return "\n".join(lines)


class ChatService:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    @staticmethod
    def _set_status(response_messages, content):
        idx = find_msg_idx(response_messages, "status")
        if idx is None:
            response_messages.insert(0, make_message(content, title="状态", node="status"))
        else:
            response_messages[idx]["content"] = content

    @staticmethod
    def _clear_status(response_messages):
        idx = find_msg_idx(response_messages, "status")
        if idx is not None:
            response_messages.pop(idx)

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]
        title = SYSTEM_NODE_CONFIG[node]["title"]
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer
        self._set_status(response_messages, "正在理解问题并检索证据...")

        idx = find_msg_idx(response_messages, node)
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        data = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_question", "")
        if data.get("needs_clarification") and clarification.strip():
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, title="需要澄清", node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _surface_interrupted_clarification(self, config_obj, response_messages):
        final_state = self.rag_system.agent_graph.get_state(config_obj)
        values = final_state.values or {}
        clarification = str(values.get("clarification_question", "") or "").strip()
        if final_state.next and clarification:
            self._clear_status(response_messages)
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, title="需要澄清", node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification
            return True
        return False

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        self._set_status(response_messages, "正在搜索当前工作区...")
        for tool_call in chunk.tool_calls:
            if tool_call.get("name") not in VISIBLE_TOOL_NAMES:
                continue
            if tool_call.get("id") and tool_call["id"] not in active_tool_calls:
                display_name = TOOL_DISPLAY_NAMES.get(tool_call["name"], tool_call["name"])
                response_messages.append(
                    make_message(f"正在运行“{display_name}”...", title=f"工具：{display_name}")
                )
                active_tool_calls[tool_call["id"]] = len(response_messages) - 1

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        idx = active_tool_calls.get(chunk.tool_call_id)
        if idx is not None:
            formatted_content = format_tool_preview(chunk.content)
            preview = formatted_content[:300]
            suffix = "\n..." if len(formatted_content) > 300 else ""
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, node, response_messages):
        self._clear_status(response_messages)
        if node in {"reflect_answer", "verify_answer"}:
            idx = next(
                (
                    i
                    for i, message in enumerate(response_messages)
                    if message.get("metadata", {}).get("node")
                    in {"aggregate_answers", "close_reading", "compare_papers", "literature_review", "reflect_answer", "verify_answer"}
                ),
                None,
            )
            if idx is None:
                response_messages.append(make_message("", title="最终答案", node="reflect_answer"))
                idx = len(response_messages) - 1
            else:
                previous_node = response_messages[idx].get("metadata", {}).get("node")
                response_messages[idx]["metadata"] = {"title": "最终答案", "node": "reflect_answer"}
                if previous_node != "reflect_answer" and response_messages[idx]["content"]:
                    response_messages[idx]["content"] = ""
            response_messages[idx]["content"] += chunk.content
            return

        idx = find_msg_idx(response_messages, node) if node else None
        if idx is None:
            response_messages.append(make_message("", node=node))
            idx = len(response_messages) - 1
        response_messages[idx]["content"] += chunk.content

    def chat(self, message, history, workspace_id, focus_paper_id=None):
        if not self.rag_system.agent_graph:
            yield "系统尚未完成初始化。"
            return

        config_obj = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config_obj)
        current_values = current_state.values or {}
        self.rag_system.sync_workspace_context(current_values.get("workspace_context"))
        self.rag_system.set_workspace_context(workspace_id, focus_paper_id=focus_paper_id)
        config_obj = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config_obj)
        working_memory = self.rag_system.workspace_memory.load_working_memory_snapshot(workspace_id)

        try:
            state_update = {
                "messages": [HumanMessage(content=message.strip())],
                "workspace_context": self.rag_system.get_workspace_context()["workspace_context"],
                "working_memory": working_memory,
            }
            if current_state.next:
                self.rag_system.agent_graph.update_state(config_obj, state_update)
                stream_input = None
            else:
                stream_input = state_update

            response_messages = []
            active_tool_calls = {}
            system_node_buffer = {}
            self._set_status(response_messages, "正在分析问题...")
            yield response_messages

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=config_obj, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer)
                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls)
                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls)
                elif isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                yield response_messages

            if self._surface_interrupted_clarification(config_obj, response_messages):
                final_state = self.rag_system.agent_graph.get_state(config_obj)
                self.rag_system.sync_workspace_context((final_state.values or {}).get("workspace_context"))
                yield response_messages
            elif find_msg_idx(response_messages, "status") is not None:
                self._clear_status(response_messages)
                final_state = self.rag_system.agent_graph.get_state(config_obj)
                self.rag_system.sync_workspace_context((final_state.values or {}).get("workspace_context"))
                yield response_messages
            else:
                final_state = self.rag_system.agent_graph.get_state(config_obj)
                self.rag_system.sync_workspace_context((final_state.values or {}).get("workspace_context"))

        except Exception as exc:
            yield f"发生错误：{exc}"

    def clear_session(self):
        self.rag_system.reset_thread()
        self.rag_system.observability.flush()


ChatInterface = ChatService
