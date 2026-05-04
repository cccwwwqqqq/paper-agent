from __future__ import annotations

import logging

from agentic_rag.settings import Settings

logger = logging.getLogger(__name__)


class Observability:
    def __init__(self, settings: Settings):
        self._enabled = settings.langfuse_enabled
        self._handler = None
        self._client = None

        if not self._enabled:
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning("Langfuse enabled but API keys are missing; skipping initialization.")
            self._enabled = False
            return

        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            self._client = get_client()

            if self._client.auth_check():
                logger.info("Langfuse client is authenticated and ready.")
            else:
                logger.warning("Langfuse authentication failed. Disabling observability.")
                self._enabled = False
                return

            self._handler = CallbackHandler()
        except Exception as exc:
            logger.warning("Could not initialize Langfuse: %s", exc)
            self._enabled = False

    def get_handler(self):
        return self._handler

    def flush(self):
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass

