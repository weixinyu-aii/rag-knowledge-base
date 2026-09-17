import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rag_knowledge_base.api import create_app
from rag_knowledge_base.config import Settings
from rag_knowledge_base.models import AnswerResult, Citation, DocumentMeta


class FakeSessions:
    def __init__(self):
        self.messages = []

    def get(self, session_id, limit=50):
        return self.messages[-limit:]


class FakeService:
    def __init__(self):
        self.sessions = FakeSessions()
        self.documents = []

    def health(self):
        return {"status": "ok", "index_ready": True, "documents": 0}

    def list_documents(self):
        return self.documents

    def ingest_bytes(self, filename, content):
        document = DocumentMeta(
            id="abc",
            name=filename,
            stored_name="abc.txt",
            extension=Path(filename).suffix,
            size=len(content),
            sha256="a" * 64,
            created_at="2026-01-01T00:00:00+00:00",
            chunk_count=1,
        )
        self.documents.append(document)
        return document, True

    def delete_document(self, doc_id):
        return False

    def rebuild_index(self):
        return {"documents": 0, "chunks": 0}

    def search(self, question, top_k=None, filters=None):
        return []

    def ask(self, question, session_id="default", top_k=None, filters=None):
        return AnswerResult(
            answer="测试回答 [1]",
            citations=[
                Citation(
                    index=1,
                    chunk_id="abc:0",
                    source="handbook.md",
                    page=1,
                    score=0.9,
                    snippet="测试片段",
                )
            ],
            question=question,
        )

    def clear_session(self, session_id):
        self.sessions.messages = []


class ApiTests(unittest.TestCase):
    def make_settings(self, directory):
        return Settings.from_env(
            {"RKB_DATA_DIR": directory, "RKB_CORS_ORIGINS": ""},
            load_file=False,
        )

    def test_health_and_query(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(FakeService(), self.make_settings(directory))
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                response = client.post("/api/v1/query", json={"question": "年假多少天"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["answer"], "测试回答 [1]")

    def test_upload_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(FakeService(), self.make_settings(directory))
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/documents",
                    files={"file": ("handbook.txt", b"ten days", "text/plain")},
                )
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json()["name"], "handbook.txt")

    def test_api_key_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.make_settings(directory)
            settings.api_key = "secret"
            app = create_app(FakeService(), settings)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/api/v1/documents").status_code, 401)
                response = client.get(
                    "/api/v1/documents",
                    headers={"X-API-Key": "secret"},
                )
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
