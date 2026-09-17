import tempfile
import unittest
from pathlib import Path

from rag_knowledge_base.manifest import DocumentManifest
from rag_knowledge_base.models import DocumentMeta
from rag_knowledge_base.sessions import SQLiteSessionStore


class PersistenceTests(unittest.TestCase):
    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = DocumentManifest(Path(directory) / "manifest.json")
            document = DocumentMeta(
                id="abc",
                name="handbook.md",
                stored_name="abc.md",
                extension=".md",
                size=10,
                sha256="a" * 64,
                created_at="2026-01-01T00:00:00+00:00",
                chunk_count=2,
            )
            manifest.add(document)
            loaded = DocumentManifest(Path(directory) / "manifest.json")
            self.assertEqual(loaded.get("abc").chunk_count, 2)
            self.assertEqual(loaded.find_by_hash("a" * 64).name, "handbook.md")

    def test_sqlite_session_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.append_many("s1", [("user", "问题"), ("assistant", "回答")])
            self.assertEqual(store.get("s1"), [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答"},
            ])
            store.clear("s1")
            self.assertEqual(store.get("s1"), [])


if __name__ == "__main__":
    unittest.main()
