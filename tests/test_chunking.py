import unittest

from rag_knowledge_base.chunking import normalize_text, split_text
from rag_knowledge_base.models import LoadedPage
from rag_knowledge_base.chunking import chunk_pages


class ChunkingTests(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text(" 第一行  \n\n\n  第二行  "), "第一行\n\n第二行")

    def test_split_text_respects_size(self):
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。"
        chunks = split_text(text, chunk_size=12, overlap=3)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))

    def test_overlap_uses_complete_sentence(self):
        text = "第一句话。第二句话。第三句话。"
        chunks = split_text(text, chunk_size=10, overlap=5)
        self.assertTrue(chunks[1].startswith("第二句话") or chunks[1].startswith("第三句话"))

    def test_chunk_pages_adds_traceable_metadata(self):
        pages = [LoadedPage(text="员工每年享有十天年假。", page=3)]
        chunks = chunk_pages(
            pages,
            doc_id="doc123",
            source="handbook.pdf",
            chunk_size=20,
            overlap=5,
        )
        self.assertEqual(chunks[0].id, "doc123:0")
        self.assertEqual(chunks[0].metadata["source"], "handbook.pdf")
        self.assertEqual(chunks[0].metadata["page"], 3)


if __name__ == "__main__":
    unittest.main()
