from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

from agentic_rag.bootstrap import build_runtime
from agentic_rag.settings import Settings, get_settings
from agentic_rag.ui.css import custom_css
from agentic_rag.ui.gradio_app import CUSTOM_HEAD, create_gradio_ui


def _ensure_localhost_bypasses_proxy() -> None:
    bypass_hosts = {"127.0.0.1", "localhost", "::1"}
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        values = {item.strip() for item in existing.split(",") if item.strip()}
        os.environ[env_name] = ",".join(sorted(values | bypass_hosts))


class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):  # pragma: no cover - defensive logging filter
        return "Failed to detach context" not in record.getMessage()


class _SingleInstanceGuard:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._handle = None

    def acquire(self) -> None:
        import portalocker

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            portalocker.lock(self._handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(_duplicate_instance_message(self.lock_path)) from exc

        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        atexit.register(self.release)

    def release(self) -> None:
        if self._handle is None:
            return

        try:
            import portalocker

            self._handle.seek(0)
            self._handle.truncate()
            portalocker.unlock(self._handle)
        except Exception:
            pass
        finally:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None


def _duplicate_instance_message(lock_path: Path) -> str:
    return (
        "Another `agentic_rag.app` instance is already running for this project.\n"
        "Open http://127.0.0.1:7860 if the UI is already up, or stop the other Python process before starting a new one.\n"
        f"Lock file: {lock_path}"
    )


def _instance_lock_path(settings: Settings) -> Path:
    return Path(settings.data_dir) / ".app.lock"


def _create_single_instance_guard(settings: Settings) -> _SingleInstanceGuard:
    guard = _SingleInstanceGuard(_instance_lock_path(settings))
    guard.acquire()
    return guard


def _is_qdrant_lock_error(exc: Exception) -> bool:
    return "already accessed by another instance of Qdrant client" in str(exc)


def main() -> None:
    _ensure_localhost_bypasses_proxy()
    logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())
    settings = get_settings()
    guard = _create_single_instance_guard(settings)

    try:
        runtime = build_runtime(settings=settings)
        demo = create_gradio_ui(runtime)
        demo.launch(css=custom_css, head=CUSTOM_HEAD)
    except RuntimeError as exc:
        if _is_qdrant_lock_error(exc):
            raise SystemExit(_duplicate_instance_message(_instance_lock_path(settings))) from exc
        raise
    finally:
        guard.release()


if __name__ == "__main__":
    main()
