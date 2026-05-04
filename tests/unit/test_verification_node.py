import unittest

from agentic_rag.agents.nodes import finalize_interaction, metadata_query_response, verify_answer, workspace_inventory_response
from agentic_rag.agents.schemas import AnswerVerification


class _FakeStructuredLlm:
    def __init__(self, payload):
        self.payload = payload
        self.invoke_calls = 0
        self.last_messages = None

    def with_structured_output(self, schema, method="function_calling"):
        return self

    def invoke(self, messages):
        self.invoke_calls += 1
        self.last_messages = messages
        return AnswerVerification(**self.payload)


class _FakeWorkspaceMemory:
    def __init__(self, papers=None):
        self.papers = papers or []
        self.records = []

    def list_papers(self, workspace_id):
        return self.papers

    def record_interaction(self, workspace_id, payload):
        self.records.append((workspace_id, payload))


class VerificationNodeTests(unittest.TestCase):
    def test_verify_answer_marks_supported_answer_as_pass(self):
        state = {
            "final_answer": "The method uses blockchain for auditing.",
            "originalQuery": "What is blockchain used for?",
            "retrieved_parent_chunks": [
                {
                    "section": "System Model",
                    "pages": "2-3",
                    "source": "paper.pdf",
                    "content_type": "paragraph",
                    "content": "Blockchain is used to audit access actions.",
                }
            ],
        }

        result = verify_answer(state, _FakeStructuredLlm({"verification_status": "pass", "revised_answer": "The method uses blockchain for auditing."}))

        self.assertEqual(result["verification_status"], "pass")
        self.assertIn("Sources", result["final_answer"])

    def test_verify_answer_can_downgrade_claims(self):
        state = {
            "final_answer": "The paper proves perfect privacy.",
            "originalQuery": "What privacy guarantees are proven?",
            "retrieved_parent_chunks": [
                {
                    "section": "Security Model",
                    "pages": "5-6",
                    "source": "paper.pdf",
                    "content_type": "paragraph",
                    "content": "The excerpt discusses privacy goals but does not prove perfect privacy.",
                }
            ],
        }

        result = verify_answer(state, _FakeStructuredLlm({"verification_status": "downgrade", "revised_answer": "The excerpts discuss privacy goals, but they do not establish perfect privacy."}))

        self.assertEqual(result["verification_status"], "downgrade")
        self.assertIn("do not establish perfect privacy", result["final_answer"])

    def test_verify_answer_short_circuits_valid_fallback_without_llm_call(self):
        llm = _FakeStructuredLlm({"verification_status": "pass", "revised_answer": "unused"})
        state = {
            "intent_type": "single_doc_close_reading",
            "final_answer": "The paper does not report any BLEU score.",
            "originalQuery": "In the PM-ABE paper, what BLEU score is reported?",
            "workspace_context": {"focus_paper_id": "paper-1"},
            "retrieved_parent_chunks": [
                {
                    "section": "Performance Analysis",
                    "pages": "8-9",
                    "source": "paper.pdf",
                    "content_type": "paragraph",
                    "content": "The experiments evaluate encryption and decryption time.",
                }
            ],
        }

        result = verify_answer(state, llm)

        self.assertEqual(result["verification_status"], "downgrade")
        self.assertEqual(
            result["final_answer"],
            "I could not find enough supporting passages in the selected paper to answer that question.",
        )
        self.assertEqual(llm.invoke_calls, 0)

    def test_verify_answer_fast_passes_supported_single_doc_answer_without_llm_call(self):
        llm = _FakeStructuredLlm({"verification_status": "downgrade", "revised_answer": "unused"})
        state = {
            "intent_type": "single_doc_close_reading",
            "final_answer": "Blockchain is used for auditing access actions.",
            "originalQuery": "What is blockchain used for?",
            "workspace_context": {"focus_paper_id": "paper-1"},
            "retrieved_parent_chunks": [
                {
                    "section": "Blockchain",
                    "pages": "2-3",
                    "source": "paper.pdf",
                    "content_type": "paragraph",
                    "content": "Blockchain is used to audit access actions and share search information.",
                }
            ],
        }

        result = verify_answer(state, llm)

        self.assertEqual(result["verification_status"], "pass")
        self.assertIn("Sources", result["final_answer"])
        self.assertEqual(llm.invoke_calls, 0)

    def test_verify_answer_multidoc_uses_structured_verifier_instead_of_keyword_downgrade(self):
        llm = _FakeStructuredLlm(
            {
                "verification_status": "downgrade",
                "revised_answer": "The supplied excerpts support only CEDC-ABE's blockchain auditing role, not a shared blockchain claim for both papers.",
            }
        )
        state = {
            "intent_type": "cross_doc_comparison",
            "final_answer": "Both papers use blockchain to audit receiver access.",
            "originalQuery": "Compare how CEDC-ABE and PM-ABE use blockchain for receiver authorization.",
            "retrieved_parent_chunks": [
                {
                    "paper_id": "cedc",
                    "section": "Blockchain",
                    "pages": "4-5",
                    "source": "cedc.pdf",
                    "content_type": "paragraph",
                    "content": "CEDC-ABE uses blockchain to audit access actions.",
                },
                {
                    "paper_id": "pmabe",
                    "section": "System Model",
                    "pages": "3-4",
                    "source": "pmabe.pdf",
                    "content_type": "paragraph",
                    "content": "PM-ABE discusses data owners and data receivers.",
                },
            ],
        }

        result = verify_answer(state, llm)

        self.assertEqual(result["verification_status"], "downgrade")
        self.assertIn("support only CEDC-ABE", result["final_answer"])
        self.assertEqual(llm.invoke_calls, 1)
        self.assertIn("Paper: cedc", llm.last_messages[-1].content)
        self.assertIn("Paper: pmabe", llm.last_messages[-1].content)

    def test_verify_answer_multidoc_without_evidence_downgrades_without_llm_call(self):
        llm = _FakeStructuredLlm({"verification_status": "pass", "revised_answer": "unused"})
        state = {
            "intent_type": "literature_review",
            "final_answer": "The workspace papers report a shared ImageNet result.",
            "originalQuery": "Review ImageNet results across the workspace papers.",
            "retrieved_parent_chunks": [],
        }

        result = verify_answer(state, llm)

        self.assertEqual(result["verification_status"], "downgrade")
        self.assertIn("supporting passages", result["final_answer"])
        self.assertEqual(llm.invoke_calls, 0)

    def test_finalize_interaction_records_verified_status(self):
        memory = _FakeWorkspaceMemory()
        state = {
            "workspace_context": {"workspace_id": "demo"},
            "intent_type": "single_doc_close_reading",
            "task_intent": "single_paper_qa",
            "originalQuery": "What is blockchain used for?",
            "retrieved_paper_ids": ["paper-1"],
            "retrieved_sections": ["Blockchain"],
            "verification_status": "pass",
        }

        result = finalize_interaction(state, memory, default_workspace_id="default")

        self.assertEqual(result["verification_status"], "pass")
        self.assertEqual(memory.records[0][0], "demo")
        self.assertEqual(memory.records[0][1]["verification_status"], "pass")
        self.assertEqual(memory.records[0][1]["paper_ids"], ["paper-1"])

    def test_deterministic_metadata_and_inventory_mark_verification_not_applicable(self):
        memory = _FakeWorkspaceMemory(
            papers=[{"paper_id": "paper-1", "title": "Paper One", "source_name": "paper-one.md"}]
        )
        state = {
            "workspace_context": {"workspace_id": "demo"},
            "originalQuery": "What papers are available in the current workspace?",
            "intent_type": "general_retrieval",
        }

        inventory = workspace_inventory_response(state, memory, default_workspace_id="default")
        metadata = metadata_query_response(
            {**state, "originalQuery": "Give the titles of the papers."},
            memory,
            default_workspace_id="default",
        )

        self.assertEqual(inventory["verification_status"], "not_applicable")
        self.assertEqual(metadata["verification_status"], "not_applicable")
        self.assertEqual(memory.records[0][1]["verification_status"], "not_applicable")
        self.assertEqual(memory.records[1][1]["verification_status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
