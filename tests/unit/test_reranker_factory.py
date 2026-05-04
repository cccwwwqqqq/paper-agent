import json
import unittest
from unittest import mock

from agentic_rag.models.reranker_factory import SiliconFlowReranker


class _FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class SiliconFlowRerankerTests(unittest.TestCase):
    def test_rerank_sends_expected_payload_and_parses_scores(self):
        reranker = SiliconFlowReranker(
            model="BAAI/bge-reranker-v2-m3",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1/rerank",
        )

        with mock.patch("urllib.request.urlopen", return_value=_FakeHttpResponse({"results": [{"index": 1, "relevance_score": 0.8}]})) as mocked:
            results = reranker.rerank("hello", ["doc-a", "doc-b"], top_n=1)

        request_obj = mocked.call_args[0][0]
        payload = json.loads(request_obj.data.decode("utf-8"))
        self.assertEqual(payload["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(payload["query"], "hello")
        self.assertEqual(payload["documents"], ["doc-a", "doc-b"])
        self.assertEqual(payload["top_n"], 1)
        self.assertEqual(results[0].index, 1)
        self.assertEqual(results[0].relevance_score, 0.8)


if __name__ == "__main__":
    unittest.main()
