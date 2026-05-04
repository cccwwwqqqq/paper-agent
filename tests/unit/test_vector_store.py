import unittest

from agentic_rag.settings import load_settings
from agentic_rag.storage.vector_store import VectorStoreManager


class VectorStoreFilterTests(unittest.TestCase):
    def test_get_filter_includes_workspace_only(self):
        settings = load_settings(use_env=False)
        manager = object.__new__(VectorStoreManager)
        manager.settings = settings

        qdrant_filter = manager.get_filter(workspace_id="demo")

        self.assertEqual(len(qdrant_filter.must), 1)
        self.assertEqual(qdrant_filter.must[0].key, "metadata.workspace_id")
        self.assertEqual(qdrant_filter.must[0].match.value, "demo")

    def test_get_filter_includes_optional_paper_and_section(self):
        settings = load_settings(use_env=False)
        manager = object.__new__(VectorStoreManager)
        manager.settings = settings

        qdrant_filter = manager.get_filter(workspace_id="demo", paper_id="paper-1", section="Method")

        self.assertEqual(len(qdrant_filter.must), 3)
        self.assertEqual([condition.key for condition in qdrant_filter.must], [
            "metadata.workspace_id",
            "metadata.paper_id",
            "metadata.section",
        ])
        self.assertEqual(qdrant_filter.must[1].match.value, "paper-1")
        self.assertEqual(qdrant_filter.must[2].match.value, "Method")


if __name__ == "__main__":
    unittest.main()
