import unittest

from agentic_rag.agents.retriever_policy import RetrieverPolicy


class RetrieverPolicyTests(unittest.TestCase):
    def test_from_context_reads_retrieval_plan_fields(self):
        policy = RetrieverPolicy.from_context(
            {
                "workspace_id": "paper-test",
                "intent_type": "cross_doc_comparison",
                "focus_paper_id": "paper-a",
                "knowledge_scope": "workspace_documents",
                "retrieval_plan": {
                    "mode": "per_paper_compare",
                    "paper_ids": ["paper-a", "paper-b"],
                    "section_hints": ["performance", "evaluation"],
                    "content_type_hints": ["table", "figure"],
                    "per_paper_limit": 2,
                    "global_limit": 6,
                },
            },
            default_workspace_id="default",
            default_knowledge_scope="workspace_documents",
        )

        self.assertEqual(policy.workspace_id, "paper-test")
        self.assertEqual(policy.mode, "per_paper_compare")
        self.assertEqual(policy.effective_paper_ids(), ["paper-a", "paper-b"])
        self.assertEqual(policy.section_hints, ["performance", "evaluation"])
        self.assertEqual(policy.content_type_hints, ["table", "figure"])
        self.assertEqual(policy.per_paper_limit, 2)
        self.assertEqual(policy.global_limit, 6)

    def test_effective_paper_ids_falls_back_to_focus_paper(self):
        policy = RetrieverPolicy(
            workspace_id="demo",
            focus_paper_id="paper-focus",
            paper_ids=[],
        )

        self.assertEqual(policy.effective_paper_ids(), ["paper-focus"])


if __name__ == "__main__":
    unittest.main()
