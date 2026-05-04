from agentic_rag.bootstrap import build_runtime
from agentic_rag.ui.gradio_app import CUSTOM_HEAD, create_gradio_ui as _create_gradio_ui


def create_gradio_ui():
    return _create_gradio_ui(build_runtime())
