import unittest
from unittest import mock

from agentic_rag.models.embedding_factory import create_dense_embeddings
from agentic_rag.models.llm_factory import create_llm
from agentic_rag.models.reranker_factory import SiliconFlowReranker, create_reranker
from agentic_rag.settings import load_settings


class ModelFactoryValidationTests(unittest.TestCase):
    def test_create_llm_requires_openai_key(self):
        settings = load_settings(use_env=False)
        object.__setattr__(settings, "llm_provider", "openai")
        object.__setattr__(settings, "openai_api_key", "")

        with self.assertRaisesRegex(ValueError, "OpenAI API key is missing"):
            create_llm(settings)

    def test_create_llm_requires_deepseek_key(self):
        settings = load_settings(use_env=False)
        object.__setattr__(settings, "llm_provider", "deepseek")
        object.__setattr__(settings, "deepseek_api_key", "")

        with self.assertRaisesRegex(ValueError, "DeepSeek API key is missing"):
            create_llm(settings)

    def test_create_embeddings_requires_api_key(self):
        settings = load_settings(use_env=False)
        object.__setattr__(settings, "embedding_provider", "openai")
        object.__setattr__(settings, "embedding_api_key", "")

        with self.assertRaisesRegex(ValueError, "Embedding API key is missing"):
            create_dense_embeddings(settings)

    def test_create_embeddings_supports_siliconflow_alias(self):
        settings = load_settings(use_env=False)
        object.__setattr__(settings, "embedding_provider", "siliconflow")
        object.__setattr__(settings, "embedding_api_key", "test-key")
        object.__setattr__(settings, "embedding_base_url", "https://api.siliconflow.cn/v1")
        object.__setattr__(settings, "embedding_model", "BAAI/bge-m3")

        with mock.patch("langchain_openai.OpenAIEmbeddings", return_value="embeddings-client") as mocked:
            result = create_dense_embeddings(settings)

        self.assertEqual(result, "embeddings-client")
        self.assertEqual(mocked.call_args.kwargs["model"], "BAAI/bge-m3")
        self.assertEqual(mocked.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(mocked.call_args.kwargs["base_url"], "https://api.siliconflow.cn/v1")

    def test_create_reranker_returns_siliconflow_client(self):
        settings = load_settings(use_env=False)
        object.__setattr__(settings, "rerank_api_key", "test-key")

        reranker = create_reranker(settings)

        self.assertIsInstance(reranker, SiliconFlowReranker)
        self.assertTrue(reranker.is_available)


if __name__ == "__main__":
    unittest.main()
