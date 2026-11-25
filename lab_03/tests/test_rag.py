import unittest

from questions import BANK, BOUNDARY, QUESTIONS
from rag import Page, PDFLoader, RAG, REWRITE_PROMPT, binary_index, split_pages

from experiments import PDF_PATH


class RAGTests(unittest.TestCase):
    def test_split_preserves_page_and_offset(self):
        pages = [Page("sample.pdf", 1, "abcdefgh"), Page("sample.pdf", 2, "wxyz")]
        chunks = split_pages(pages, 5, 2)
        self.assertEqual(
            [(c.id, c.page, c.start, c.text) for c in chunks],
            [(1, 1, 0, "abcde"), (2, 1, 3, "defgh"), (3, 2, 0, "wxyz")],
        )
        with self.assertRaises(ValueError):
            split_pages(pages, 5, 5)

    def test_binary_query_without_terms_returns_no_hits(self):
        chunks = split_pages([Page("sample.pdf", 1, "alpha alpha")], 30, 0)
        index = binary_index(chunks)
        self.assertEqual(index.search("beta"), [])
        self.assertAlmostEqual(index.search("Alpha")[0].score, 1)

    def test_session_window_isolation_clear_and_fresh_context(self):
        chunks = split_pages([
            Page("sample.pdf", 1, "alpha source"), Page("sample.pdf", 2, "beta source"),
        ], 30, 0)
        calls = []

        def generate(prompt):
            calls.append(prompt)
            if prompt.startswith(REWRITE_PROMPT):
                return "beta"
            return "answer"

        rag = RAG(binary_index(chunks), generate)
        first = rag.ask("alpha", session="first")
        self.assertEqual(first.history, ())
        self.assertEqual(first.hits[0].chunk.page, 1)
        second = rag.ask("What about it?", session="first")
        self.assertEqual(second.query, "beta")
        self.assertEqual(second.hits[0].chunk.page, 2)
        self.assertIn("beta source", calls[-1])
        self.assertNotIn("alpha source", calls[-1])
        self.assertEqual(rag.ask("alpha", session="other").history, ())
        self.assertEqual(rag.ask("alpha").history, ())
        rag.ask("turn three", session="first")
        rag.ask("turn four", session="first")
        fifth = rag.ask("turn five", session="first")
        self.assertEqual([q for q, _ in fifth.history], ["What about it?", "turn three", "turn four"])
        rag.clear("first")
        self.assertEqual(rag.ask("alpha", session="first").history, ())
        self.assertEqual(len(rag.ask("alpha", session="other").history), 1)

    def test_reference_quotes_and_boundary_selection(self):
        pages = PDFLoader(PDF_PATH).load()
        self.assertEqual(len(pages), 68)
        self.assertEqual(len(BANK), 10)
        for question in QUESTIONS.values():
            for page, quote in question.sources:
                with self.subTest(question=question.id, page=page):
                    self.assertIn(quote, pages[page - 1].text)
        for qid, page, boundary in BOUNDARY:
            quote = next(quote for number, quote in QUESTIONS[qid].sources if number == page)
            start = pages[page - 1].text.index(quote)
            self.assertLess(start, boundary)
            self.assertGreater(start + len(quote), boundary)


if __name__ == "__main__":
    unittest.main()
