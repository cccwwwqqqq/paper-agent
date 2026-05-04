from __future__ import annotations

import gradio as gr

from agentic_rag.bootstrap import ApplicationRuntime

CUSTOM_HEAD = """
<script>
(() => {
  const exactTextMap = {
    "Drop files here": "将文件拖到这里",
    "Click to Upload": "点击上传",
    "No files uploaded": "暂无已上传文件",
    "Paste from clipboard": "从剪贴板粘贴",
    "Drop files here or click to upload": "将文件拖到这里或点击上传",
    "Drop files here or Click to Upload": "将文件拖到这里或点击上传",
    "Switch workspace": "切换工作区",
    "Refresh": "刷新列表",
    "Clear workspace": "清空工作区",
    "Add documents": "添加文档",
    "Send": "发送",
    "Stop": "停止",
    "Undo": "撤销",
    "Retry": "重试",
    "Clear": "清空",
    "Remove": "移除",
    "Delete": "删除",
    "Copy": "复制",
    "Edit": "编辑",
    "or": "或"
  };

  const placeholderMap = {
    "Type a message...": "请输入你的问题...",
    "Ask a question about the active workspace...": "请输入关于当前工作区的问题..."
  };

  const replaceText = (element) => {
    const text = element.textContent?.trim();
    if (text && exactTextMap[text] && element.children.length === 0) {
      element.textContent = exactTextMap[text];
    }
  };

  const translateElement = (root) => {
    if (!root) return;
    root.querySelectorAll("button, span, div, p, label").forEach(replaceText);

    root.querySelectorAll("textarea, input").forEach((element) => {
      const placeholder = element.getAttribute("placeholder");
      if (placeholder && placeholderMap[placeholder]) {
        element.setAttribute("placeholder", placeholderMap[placeholder]);
      }
      const ariaLabel = element.getAttribute("aria-label");
      if (ariaLabel && exactTextMap[ariaLabel]) {
        element.setAttribute("aria-label", exactTextMap[ariaLabel]);
      }
    });
  };

  const run = () => translateElement(document);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }

  const observer = new MutationObserver(() => translateElement(document));
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


def create_gradio_ui(runtime: ApplicationRuntime):
    settings = runtime.settings
    doc_manager = runtime.ingestion_service
    formula_refiner = runtime.formula_refinement_service
    chat_service = runtime.chat_service
    rag_system = runtime.rag_system

    def normalize_workspace(workspace_id: str) -> str:
        return (workspace_id or settings.default_workspace_id).strip() or settings.default_workspace_id

    def focus_paper_choices(workspace_id: str):
        workspace_id = normalize_workspace(workspace_id)
        papers = rag_system.workspace_memory.list_papers(workspace_id)
        choices = [("自动检测", "")]
        choices.extend((paper.get("source_name", paper["paper_id"]), paper["paper_id"]) for paper in papers)
        return gr.Dropdown(
            choices=choices,
            value="",
            show_label=False,
            container=False,
            elem_classes="compact-select",
        )

    def format_file_list(workspace_id: str):
        workspace_id = normalize_workspace(workspace_id)
        files = doc_manager.get_markdown_files(workspace_id)
        if not files:
            return f"工作区 `{workspace_id}` 中暂无已索引文档。"
        return "\n".join(files)

    def activate_workspace(workspace_draft: str):
        workspace_id = normalize_workspace(workspace_draft)
        chat_service.clear_session()
        gr.Info(f"已切换到工作区 `{workspace_id}`。")
        return (
            workspace_id,
            workspace_id,
            format_file_list(workspace_id),
            focus_paper_choices(workspace_id),
            [],
            "",
        )

    def upload_handler(files, active_workspace, progress=gr.Progress()):
        if not files:
            return None, format_file_list(active_workspace), focus_paper_choices(active_workspace)

        workspace_id = normalize_workspace(active_workspace)
        added, skipped = doc_manager.add_documents(
            files,
            workspace_id=workspace_id,
            progress_callback=lambda p, desc: progress(p, desc=desc),
        )

        if doc_manager.last_errors:
            first_error = doc_manager.last_errors[0]
            suffix = "；请查看终端日志获取完整信息。" if len(doc_manager.last_errors) > 1 else ""
            gr.Warning(f"导入失败：{first_error}{suffix}")
        else:
            gr.Info(f"已新增 {added} 个文档，跳过 {skipped} 个。")
        return None, format_file_list(workspace_id), focus_paper_choices(workspace_id)

    def clear_handler(active_workspace):
        workspace_id = normalize_workspace(active_workspace)
        doc_manager.clear_all(workspace_id)
        gr.Info(f"已清空工作区 `{workspace_id}` 的全部索引内容。")
        return format_file_list(workspace_id), focus_paper_choices(workspace_id)

    def refresh_handler(active_workspace):
        workspace_id = normalize_workspace(active_workspace)
        return format_file_list(workspace_id), focus_paper_choices(workspace_id)

    def refine_formula_handler(active_workspace, focus_paper_id, progress=gr.Progress()):
        workspace_id = normalize_workspace(active_workspace)
        paper_id = focus_paper_id or None
        progress(0.05, desc="Finding formula-heavy pages")
        report = formula_refiner.refine_workspace(
            workspace_id,
            paper_id=paper_id,
            progress_callback=lambda p, desc: progress(p, desc=desc),
        )
        progress(1.0, desc="Formula refinement complete")
        if not report["candidate_count"]:
            gr.Info("No formula placeholders found for refinement.")
            return format_file_list(workspace_id), focus_paper_choices(workspace_id)
        refined_pages = sum(len(item.get("pages_refined", [])) for item in report["reports"])
        merged_pages = sum(len(item.get("merged_pages", [])) for item in report["reports"])
        gr.Info(
            f"Formula refinement complete: {report['candidate_count']} document(s), "
            f"{refined_pages} refined page(s), {merged_pages} merged page(s)."
        )
        return format_file_list(workspace_id), focus_paper_choices(workspace_id)

    def chat_handler(msg, hist, active_workspace, focus_paper_id):
        workspace_id = normalize_workspace(active_workspace)
        focus = focus_paper_id or None
        for chunk in chat_service.chat(msg, hist, workspace_id, focus_paper_id=focus):
            yield chunk

    def clear_chat_handler():
        chat_service.clear_session()

    with gr.Blocks(title="文献工作区助手", fill_width=True) as demo:
        active_workspace_state = gr.State(settings.default_workspace_id)

        with gr.Column(elem_classes="app-shell"):
            gr.Markdown(
                """
                <div class="app-hero">
                  <p class="app-eyebrow">文献工作区</p>
                  <h1>文献工作区助手</h1>
                  <p class="app-subtitle">围绕工作区管理、文档导入、精读分析与对话检索，提供一套更清晰的文献工作台。</p>
                </div>
                """
            )

            with gr.Row(elem_classes="workspace-grid"):
                with gr.Column(elem_classes="panel panel-strong"):
                    gr.Markdown(
                        """
                        <div class="panel-head">
                          <h3>工作区切换</h3>
                          <p>输入工作区名称后确认，即可切换当前上下文与会话状态。</p>
                        </div>
                        """
                    )
                    workspace_draft = gr.Textbox(
                        value=settings.default_workspace_id,
                        show_label=False,
                        container=False,
                        placeholder="输入工作区名称，例如：默认工作区、论文精读、对比实验",
                        elem_classes="compact-input",
                    )
                    confirm_workspace_btn = gr.Button("切换工作区", variant="primary", elem_classes="action-button")

                with gr.Column(elem_classes="panel panel-muted"):
                    gr.Markdown(
                        """
                        <div class="panel-head">
                          <h3>当前上下文</h3>
                          <p>聚焦论文可将检索范围收敛到单篇文献，适合做精读分析与公式追踪。</p>
                        </div>
                        """
                    )
                    gr.Markdown(
                        """
                        <div class="field-meta">
                          <label>当前工作区</label>
                        </div>
                        """
                    )
                    active_workspace_display = gr.Textbox(
                        value=settings.default_workspace_id,
                        show_label=False,
                        container=False,
                        interactive=False,
                        elem_classes="compact-output",
                    )
                    gr.Markdown(
                        """
                        <div class="field-meta">
                          <label>聚焦论文</label>
                        </div>
                        """
                    )
                    focus_paper = gr.Dropdown(
                        choices=[("自动检测", "")],
                        value="",
                        show_label=False,
                        container=False,
                        elem_classes="compact-select",
                    )

            with gr.Tabs(elem_classes="workspace-tabs"):
                with gr.Tab("文档管理", elem_id="doc-management-tab"):
                    with gr.Row(equal_height=False, elem_classes="content-grid"):
                        with gr.Column(elem_classes="panel panel-upload"):
                            gr.Markdown(
                                """
                                <div class="panel-head">
                                  <h2>添加文档</h2>
                                  <p>上传 PDF 或 Markdown 文件到当前工作区，系统会自动切分、建索引并登记到工作区目录。</p>
                                </div>
                                """
                            )
                            files_input = gr.File(
                                label="上传 PDF 或 Markdown 文档",
                                file_count="multiple",
                                type="filepath",
                                height=240,
                                elem_id="literature-upload",
                            )
                            add_btn = gr.Button("添加文档", variant="primary", size="md", elem_classes="action-button")

                        with gr.Column(elem_classes="panel panel-library"):
                            gr.Markdown(
                                """
                                <div class="panel-head">
                                  <h2>已索引文档</h2>
                                  <p>这里会显示当前工作区下已完成索引的文档列表，可刷新或整体清空。</p>
                                </div>
                                """
                            )
                            file_list = gr.Textbox(
                                value=format_file_list(settings.default_workspace_id),
                                interactive=False,
                                lines=11,
                                max_lines=14,
                                elem_id="file-list-box",
                                show_label=False,
                            )

                            with gr.Row(elem_classes="action-row"):
                                refine_formula_btn = gr.Button("Refine formulas", size="md", elem_classes="secondary-action")
                                refresh_btn = gr.Button("刷新列表", size="md", elem_classes="secondary-action")
                                clear_btn = gr.Button("清空工作区", variant="stop", size="md", elem_classes="danger-action")

                            add_btn.click(
                                upload_handler,
                                [files_input, active_workspace_state],
                                [files_input, file_list, focus_paper],
                            )
                            refresh_btn.click(refresh_handler, [active_workspace_state], [file_list, focus_paper])
                            refine_formula_btn.click(
                                refine_formula_handler,
                                [active_workspace_state, focus_paper],
                                [file_list, focus_paper],
                            )
                            clear_btn.click(clear_handler, [active_workspace_state], [file_list, focus_paper])

                with gr.Tab("对话"):
                    with gr.Column(elem_classes="chat-shell"):
                        gr.Markdown(
                            """
                            <div class="panel-head panel-head-inline">
                              <h2>工作区对话</h2>
                              <p>可直接提问当前工作区中的文献内容，支持检索问答、精读分析、论文对比与轻量综述。</p>
                            </div>
                            """
                        )
                        chatbot = gr.Chatbot(
                            height=720,
                            placeholder=(
                                "<strong>从当前工作区开始提问。</strong><br>"
                                "<em>例如：方法核心是什么？与另一篇论文相比有何差异？作者如何设计实验？</em>"
                            ),
                            show_label=False,
                            avatar_images=None,
                            layout="panel",
                            elem_classes="chat-surface",
                        )
                        chatbot.clear(clear_chat_handler)

                        chat_input = gr.Textbox(
                            placeholder="输入你想追问的问题，例如：这篇论文的方法核心是什么？",
                            container=False,
                            submit_btn=False,
                            stop_btn=False,
                            elem_classes="chat-input",
                        )

                        gr.ChatInterface(
                            fn=chat_handler,
                            additional_inputs=[active_workspace_state, focus_paper],
                            chatbot=chatbot,
                            textbox=chat_input,
                            submit_btn="发送",
                            stop_btn="停止",
                        )

            confirm_workspace_btn.click(
                activate_workspace,
                [workspace_draft],
                [active_workspace_state, active_workspace_display, file_list, focus_paper, chatbot, chat_input],
            )

    return demo
