"""Persistent FAISS index and JSONL chunk store."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Sequence

from .exceptions import IndexNotReadyError
from .models import Chunk


class FAISSVectorIndex:
    def __init__(self, directory: Path):
        self.directory = directory
        self.index_path = directory / "vectors.faiss"
        self.chunks_path = directory / "chunks.jsonl"
        self._index = None
        self._chunks: list[Chunk] = []
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def load(self) -> bool:
        with self._lock:
            if not self.index_path.exists() or not self.chunks_path.exists():
                return False
            import faiss
            import numpy as np

            raw = np.frombuffer(self.index_path.read_bytes(), dtype="uint8")
            index = faiss.deserialize_index(raw)
            chunks = self._read_chunks()
            if index.ntotal != len(chunks):
                raise IndexNotReadyError(
                    f"FAISS vectors ({index.ntotal}) do not match chunks ({len(chunks)})"
                )
            self._index = index
            self._chunks = chunks
            return True

    def build(self, chunks: Sequence[Chunk], embedder) -> None:
        if not chunks:
            raise ValueError("Cannot build an empty index")
        with self._lock:
            vectors = self._normalized_vectors(embedder.embed_documents([c.text for c in chunks]))
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._index = index
            self._chunks = list(chunks)
            self.save()

    def add(self, chunks: Sequence[Chunk], embedder) -> None:
        if not chunks:
            return
        with self._lock:
            vectors = self._normalized_vectors(embedder.embed_documents([c.text for c in chunks]))
            if self._index is None:
                self.build(chunks, embedder)
                return
            self._index.add(vectors)
            self._chunks.extend(chunks)
            self.save()

    def clear(self) -> None:
        with self._lock:
            self._index = None
            self._chunks = []
            self._delete_files()

    def remove_document(self, doc_id: str, embedder) -> None:
        with self._lock:
            remaining = [chunk for chunk in self._chunks if chunk.metadata.get("doc_id") != doc_id]
            if len(remaining) == len(self._chunks):
                return
            if remaining:
                self.build(remaining, embedder)
            else:
                self._index = None
                self._chunks = []
                self._delete_files()

    def search(self, query_embedding: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        if self._index is None or not self._chunks:
            raise IndexNotReadyError("Vector index is not ready")
        vectors = self._normalized_vectors([query_embedding])
        scores, indices = self._index.search(vectors, min(k, len(self._chunks)))
        results: list[tuple[Chunk, float]] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            results.append((self._chunks[int(index)], float(score)))
        return results

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def save(self) -> None:
        if self._index is None:
            return
        import faiss

        self.directory.mkdir(parents=True, exist_ok=True)
        index_tmp = self.index_path.with_suffix(".faiss.tmp")
        payload = faiss.serialize_index(self._index)
        index_tmp.write_bytes(payload.tobytes())
        os.replace(index_tmp, self.index_path)

        chunks_tmp = self.chunks_path.with_suffix(".jsonl.tmp")
        with chunks_tmp.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        os.replace(chunks_tmp, self.chunks_path)

    def _read_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))
        return chunks

    def _delete_files(self) -> None:
        for path in (self.index_path, self.chunks_path):
            if path.exists():
                path.unlink()

    @staticmethod
    def _normalized_vectors(vectors: Sequence[Sequence[float]]):
        import faiss
        import numpy as np

        array = np.asarray(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        faiss.normalize_L2(array)
        return array
