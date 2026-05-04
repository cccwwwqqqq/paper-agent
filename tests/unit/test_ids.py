import unittest

from agentic_rag.utils.ids import make_safe_paper_id, resolve_paper_id


class PaperIdTests(unittest.TestCase):
    def test_make_safe_paper_id_normalizes_name(self):
        paper_id = make_safe_paper_id("My Fancy Paper v2.pdf")
        self.assertRegex(paper_id, r"^my-fancy-paper-v2-[0-9a-f]{8}$")

    def test_resolve_paper_id_preserves_safe_ids(self):
        raw = "paper-name-1a2b3c4d"
        self.assertEqual(resolve_paper_id(raw), raw)


if __name__ == "__main__":
    unittest.main()

