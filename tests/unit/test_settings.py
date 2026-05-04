import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentic_rag.settings import load_settings


class SettingsTests(unittest.TestCase):
    def test_load_settings_prefers_root_env_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_file = root / ".env"
            env_file.write_text("LLM_PROVIDER=openai\nDEFAULT_WORKSPACE_ID=demo\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_settings(root_dir=root)
            self.assertEqual(settings.llm_provider, "openai")
            self.assertEqual(settings.default_workspace_id, "demo")

    def test_env_overrides_defaults(self):
        with mock.patch.dict(os.environ, {"DEFAULT_SEARCH_LIMIT": "9"}, clear=True):
            settings = load_settings(use_env=False)
            self.assertEqual(settings.default_search_limit, 9)

    def test_docling_formula_enrichment_flag_is_loaded(self):
        with mock.patch.dict(os.environ, {"DOCLING_DO_FORMULA_ENRICHMENT": "true"}, clear=True):
            settings = load_settings(use_env=False)
            self.assertTrue(settings.docling_do_formula_enrichment)

    def test_formula_refinement_defaults_are_loaded(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_settings(use_env=False)
        self.assertEqual(settings.formula_refinement_max_pages, 0)
        self.assertEqual(settings.formula_refinement_timeout, 180)

    def test_rerank_defaults_are_loaded(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = load_settings(use_env=False)
        self.assertEqual(settings.rerank_provider, "siliconflow")
        self.assertEqual(settings.rerank_model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(settings.rerank_base_url, "https://api.siliconflow.cn/v1/rerank")
        self.assertEqual(settings.rerank_top_n, 6)
        self.assertEqual(settings.rerank_max_candidates, 24)

    def test_load_settings_falls_back_to_legacy_project_env(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_dir = root / "project"
            project_dir.mkdir()
            legacy_env = project_dir / ".env"
            legacy_env.write_text("DEFAULT_WORKSPACE_ID=legacy\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_settings(root_dir=root)

            self.assertEqual(settings.env_file, legacy_env)
            self.assertEqual(settings.default_workspace_id, "legacy")

    def test_legacy_runtime_paths_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "workspace_memory").mkdir()
            (root / "markdown_docs").mkdir()

            settings = load_settings(root_dir=root, use_env=False)

            self.assertEqual(settings.workspace_memory_path, root / "workspace_memory")
            self.assertEqual(settings.markdown_dir, root / "markdown_docs")

    def test_prefers_non_empty_legacy_path_over_empty_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data" / "docling_artifacts"
            legacy_dir = root / ".docling_artifacts"
            data_dir.mkdir(parents=True)
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "marker.txt").write_text("cached-model", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_settings(root_dir=root, use_env=False)

            self.assertEqual(settings.docling_artifacts_path, legacy_dir)


if __name__ == "__main__":
    unittest.main()
