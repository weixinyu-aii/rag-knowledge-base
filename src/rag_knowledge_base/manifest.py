"""JSON manifest for uploaded document metadata."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .models import DocumentMeta


class DocumentManifest:
    def __init__(self, path: Path):
        self.path = path
        self._documents: list[DocumentMeta] = []
        self._lock = threading.RLock()
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._documents = []
                return
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._documents = [
                DocumentMeta.from_dict(item) for item in payload.get("documents", [])
            ]

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "documents": [item.to_dict() for item in self._documents],
            }
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.path)

    def list(self) -> list[DocumentMeta]:
        with self._lock:
            return [DocumentMeta.from_dict(item.to_dict()) for item in self._documents]

    def get(self, doc_id: str) -> DocumentMeta | None:
        with self._lock:
            for item in self._documents:
                if item.id == doc_id:
                    return DocumentMeta.from_dict(item.to_dict())
        return None

    def find_by_hash(self, sha256: str) -> DocumentMeta | None:
        with self._lock:
            for item in self._documents:
                if item.sha256 == sha256:
                    return DocumentMeta.from_dict(item.to_dict())
        return None

    def add(self, document: DocumentMeta) -> None:
        with self._lock:
            self._documents = [item for item in self._documents if item.id != document.id]
            self._documents.append(document)
            self.save()

    def remove(self, doc_id: str) -> DocumentMeta | None:
        with self._lock:
            found = next((item for item in self._documents if item.id == doc_id), None)
            if found is None:
                return None
            self._documents = [item for item in self._documents if item.id != doc_id]
            self.save()
            return DocumentMeta.from_dict(found.to_dict())

    def update_chunk_counts(self, counts: dict[str, int]) -> None:
        with self._lock:
            for item in self._documents:
                item.chunk_count = counts.get(item.id, 0)
            self.save()
