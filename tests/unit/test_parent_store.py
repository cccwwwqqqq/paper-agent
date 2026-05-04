import tempfile
import unittest
from pathlib import Path

from agentic_rag.settings import load_settings
from agentic_rag.storage.parent_store import ParentStoreManager


class ParentStoreTests(unittest.TestCase):
    def test_save_and_load_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings = load_settings(root_dir=root, use_env=False)
            store = ParentStoreManager(settings)
            store.save("demo", "paper-1", "paper-1_parent_0", "content", {"section": "Intro"})
            loaded = store.load_content("demo", "paper-1", "paper-1_parent_0")
            self.assertEqual(loaded["content"], "content")
            self.assertEqual(loaded["metadata"]["section"], "Intro")


if __name__ == "__main__":
    unittest.main()

