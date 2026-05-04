import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def _ensure_localhost_bypasses_proxy():
    # Gradio probes its own local startup endpoint. On machines with a system proxy
    # but no localhost bypass, that self-check can be routed through the proxy and fail.
    bypass_hosts = {"127.0.0.1", "localhost", "::1"}
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        values = {item.strip() for item in existing.split(",") if item.strip()}
        merged = values | bypass_hosts
        os.environ[env_name] = ",".join(sorted(merged))


_ensure_localhost_bypasses_proxy()

# Suppress OTel "Failed to detach context" warning caused by generator/context interaction.
# Tracing is unaffected.
# Known bug: https://github.com/open-telemetry/opentelemetry-python/issues/2606
class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

from ui.css import custom_css
from ui.gradio_app import CUSTOM_HEAD, create_gradio_ui

if __name__ == "__main__":
    print("\n正在创建文献阅读助手...")
    demo = create_gradio_ui()
    print("\n正在启动文献阅读助手...")
    demo.launch(css=custom_css, head=CUSTOM_HEAD)
