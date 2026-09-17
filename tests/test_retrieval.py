import unittest

from rag_knowledge_base.models import Chunk
from rag_knowledge_base.retrieval import BM25, HybridRetriever, tokenize


class FakeVectorIndex:
    def __init__(self, results):
        self.results = results

    def search(self, query_embedding, k):
        return self.results[:k]


def make_chunk(chunk_id, text, source="handbook.md"):
    return Chunk(id=chunk_id, text=text, metadata={"source": source, "page": 1})


class RetrievalTests(unittest.TestCase):
    def test_tokenize_chinese_bigrams(self):
        self.assertIn("报销", tokenize("差旅报销流程"))
        self.assertIn("expense", tokenize("Expense policy"))

    def test_bm25_ranks_keyword_match_first(self):
        chunks = [
            make_chunk("1", "员工每年享有十天年假。"),
            make_chunk("2", "差旅报销需要提交发票和审批单。"),
            make_chunk("3", "公司网络密码每九十天更新一次。"),
        ]
        results = BM25(chunks).search("差旅报销需要什么", k=2)
        self.assertEqual(results[0][0].id, "2")

    def test_hybrid_retriever_fuses_vector_and_bm25(self):
        chunks = [make_chunk("1", "差旅报销材料"), make_chunk("2", "年假规则")]
        vector = FakeVectorIndex([(chunks[0], 0.9), (chunks[1], 0.2)])
        retriever = HybridRetriever(vector, mode="hybrid")
        retriever.refresh(chunks)
        results = retriever.search(
            "差旅报销",
            [1.0, 0.0],
            k=2,
            candidate_k=2,
        )
        self.assertEqual(results[0].chunk.id, "1")
        self.assertIsNotNone(results[0].vector_score)
        self.assertIsNotNone(results[0].bm25_score)

    def test_metadata_filter(self):
        chunks = [
            make_chunk("1", "公开制度", source="public.md"),
            make_chunk("2", "内部制度", source="internal.md"),
        ]
        vector = FakeVectorIndex([(chunks[0], 0.9), (chunks[1], 0.8)])
        retriever = HybridRetriever(vector, mode="vector")
        retriever.refresh(chunks)
        results = retriever.search(
            "制度",
            [1.0],
            k=2,
            candidate_k=2,
            filters={"source": "internal.md"},
        )
        self.assertEqual([item.chunk.id for item in results], ["2"])


if __name__ == "__main__":
    unittest.main()
