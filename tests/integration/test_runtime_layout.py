import unittest

from agentic_rag.settings import load_settings


class RuntimeLayoutTests(unittest.TestCase):
    def test_new_data_layout_defaults_to_data_directory(self):
        settings = load_settings(use_env=False)
        self.assertIn("data", str(settings.markdown_dir))
        self.assertIn("data", str(settings.parent_store_path))


if __name__ == "__main__":
    unittest.main()

