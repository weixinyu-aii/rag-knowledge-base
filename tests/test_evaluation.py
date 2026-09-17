import unittest

from rag_knowledge_base.evaluation import evaluate_retrieval
from rag_knowledge_base.models import Chunk, RetrievedChunk


def hit(chunk_id, source, rank_score):
    chunk = Chunk(id=chunk_id, text=f"text-{chunk_id}", metadata={"source": source, "page": 1})
    return RetrievedChunk(chunk=chunk, score=rank_score)


class EvaluationTests(unittest.TestCase):
    def test_relevant_terms_filter_chunk_level_relevance(self):
        dataset = [
            {
                "id": "q1",
                "question": "年假多少天",
                "relevant_sources": ["handbook.md"],
                "relevant_terms": ["十天"],
            }
        ]
        hits = [
            hit("1", "handbook.md", 0.9),
            RetrievedChunk(
                chunk=Chunk(
                    id="2",
                    text="年假为十天",
                    metadata={"source": "handbook.md", "page": 1},
                ),
                score=0.8,
            ),
        ]
        summary = evaluate_retrieval(dataset, lambda question, k: hits[:k], k=2)
        self.assertEqual(summary.hit_rate, 1.0)
        self.assertEqual(summary.mrr, 0.5)
        self.assertEqual(summary.precision, 0.5)

    def test_retrieval_metrics(self):
        dataset = [
            {
                "id": "q1",
                "question": "年假多少天",
                "relevant_sources": ["handbook.md"],
            }
        ]
        hits = [hit("1", "other.md", 0.9), hit("2", "handbook.md", 0.8)]
        summary = evaluate_retrieval(dataset, lambda question, k: hits[:k], k=2)
        self.assertEqual(summary.hit_rate, 1.0)
        self.assertEqual(summary.recall, 1.0)
        self.assertEqual(summary.mrr, 0.5)
        self.assertEqual(summary.precision, 0.5)


if __name__ == "__main__":
    unittest.main()
