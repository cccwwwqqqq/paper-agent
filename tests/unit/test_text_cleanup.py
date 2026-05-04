import unittest

from agentic_rag.utils.text import clean_markdown_text


class TextCleanupTests(unittest.TestCase):
    def test_repairs_common_pdf_mojibake_symbols(self):
        broken = "x \u00e2\u0088\u0088 S, \u00ce\u00bb \u00e2\u0086\u0092 y, G \u00c3\u0097 G"

        cleaned = clean_markdown_text(broken)

        self.assertEqual(cleaned, "x \u2208 S, \u03bb \u2192 y, G \u00d7 G")


if __name__ == "__main__":
    unittest.main()
