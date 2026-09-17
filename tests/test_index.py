import importlib.util
import tempfile
import unittest
from pathlib import Path

from rag_knowledge_base.index import FAISSVectorIndex
from rag_knowledge_base.models import Chunk


class FakeEmbedder:
    dimension = 4

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0, 0.0, 0.0] if "年假" in text else [0.0, 1.0, 0.0, 0.0])
        return vectors


@unittest.skipUnless(
    importlib.util.find_spec("faiss") and importlib.util.find_spec("numpy"),
    "faiss-cpu is not installed",
)
class FAISSIndexTests(unittest.TestCase):
    def test_build_save_load_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FAISSVectorIndex(Path(directory) / "index")
            chunks = [
                Chunk(id="1", text="年假为十天", metadata={"source": "handbook.md"}),
                Chunk(id="2", text="采购需要审批", metadata={"source": "handbook.md"}),
            ]
            index.build(chunks, FakeEmbedder())
            loaded = FAISSVectorIndex(Path(directory) / "index")
            self.assertTrue(loaded.load())
            results = loaded.search([1.0, 0.0, 0.0, 0.0], 1)
            self.assertEqual(results[0][0].id, "1")


if __name__ == "__main__":
    unittest.main()
