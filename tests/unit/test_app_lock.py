import tempfile
import unittest
from pathlib import Path
from unittest import mock

import portalocker

from agentic_rag.app import _create_single_instance_guard, _duplicate_instance_message, _instance_lock_path
from agentic_rag.app import _is_qdrant_lock_error
from agentic_rag.settings import load_settings


class AppLockTests(unittest.TestCase):
    def test_instance_lock_path_lives_under_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = load_settings(root_dir=Path(tmp_dir), use_env=False)

            lock_path = _instance_lock_path(settings)

            self.assertEqual(lock_path, settings.data_dir / ".app.lock")

    def test_duplicate_instance_message_mentions_default_url(self):
        message = _duplicate_instance_message(Path("data/.app.lock"))

        self.assertIn("http://127.0.0.1:7860", message)
        self.assertIn(".app.lock", message)

    def test_create_single_instance_guard_surfaces_friendly_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = load_settings(root_dir=Path(tmp_dir), use_env=False)

            with mock.patch("portalocker.lock", side_effect=portalocker.exceptions.AlreadyLocked(None, None)):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    _create_single_instance_guard(settings)

    def test_qdrant_lock_error_detector_matches_runtime_message(self):
        exc = RuntimeError("Storage folder X is already accessed by another instance of Qdrant client.")

        self.assertTrue(_is_qdrant_lock_error(exc))
        self.assertFalse(_is_qdrant_lock_error(RuntimeError("something else")))


if __name__ == "__main__":
    unittest.main()
