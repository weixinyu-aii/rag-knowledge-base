"""Lexical retrieval, reciprocal rank fusion, and optional reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Protocol, Sequence

from .models import Chunk, RetrievedChunk


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for value in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.fullmatch(value):
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


class VectorSearchIndex(Protocol):
    def search(self, query_embedding: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        ...


class BM25:
    """Small, dependency-free BM25 implementation with Chinese bigram support."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(freq.values()) for freq in self._term_frequencies]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: Counter[str] = Counter()
        for freq in self._term_frequencies:
            self._document_frequency.update(freq.keys())

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks or k < 1:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []
        total = len(self.chunks)
        scored: list[tuple[Chunk, float]] = []
        for chunk, frequencies, length in zip(
            self.chunks, self._term_frequencies, self._lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._avg_length
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def _matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(str(chunk.metadata.get(key)) == str(value) for key, value in filters.items())


class HybridRetriever:
    """Combines vector and BM25 rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_index: VectorSearchIndex,
        *,
        mode: str = "hybrid",
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ):
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("mode must be vector, bm25, or hybrid")
        self.vector_index = vector_index
        self.mode = mode
        self.rrf_k = rrf_k
        self.reranker = reranker
        self._chunks: list[Chunk] = []
        self._bm25: BM25 | None = None

    def refresh(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25(self._chunks)

    def search(
        self,
        query: str,
        query_embedding: Sequence[float] | None,
        *,
        k: int,
        candidate_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if k < 1:
            return []
        limit = max(k, candidate_k)
        vector_hits: list[tuple[Chunk, float]] = []
        bm25_hits: list[tuple[Chunk, float]] = []

        if self.mode in {"vector", "hybrid"} and query_embedding is not None:
            vector_hits = [
                item
                for item in self.vector_index.search(query_embedding, max(limit * 3, limit))
                if _matches_filters(item[0], filters)
            ][:limit]
        if self.mode in {"bm25", "hybrid"} and self._bm25 is not None:
            bm25_hits = [
                item
                for item in self._bm25.search(query, max(limit * 3, limit))
                if _matches_filters(item[0], filters)
            ][:limit]

        if self.mode == "vector":
            candidates = [
                RetrievedChunk(chunk=chunk, score=score, vector_score=score)
                for chunk, score in vector_hits[:limit]
            ]
        elif self.mode == "bm25":
            candidates = [
                RetrievedChunk(chunk=chunk, score=score, bm25_score=score)
                for chunk, score in bm25_hits[:limit]
            ]
        else:
            candidates = self._fuse(vector_hits, bm25_hits, limit)

        if self.reranker is not None and candidates:
            return self.reranker.rerank(query, candidates, k)
        return candidates[:k]

    def _fuse(
        self,
        vector_hits: Sequence[tuple[Chunk, float]],
        bm25_hits: Sequence[tuple[Chunk, float]],
        limit: int,
    ) -> list[RetrievedChunk]:
        records: dict[str, RetrievedChunk] = {}
        fused_scores: Counter[str] = Counter()

        for rank, (chunk, score) in enumerate(vector_hits, start=1):
            records[chunk.id] = RetrievedChunk(
                chunk=chunk, score=0.0, vector_score=score
            )
            fused_scores[chunk.id] += 1.0 / (self.rrf_k + rank)

        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            if chunk.id not in records:
                records[chunk.id] = RetrievedChunk(chunk=chunk, score=0.0)
            records[chunk.id].bm25_score = score
            fused_scores[chunk.id] += 1.0 / (self.rrf_k + rank)

        for chunk_id, score in fused_scores.items():
            records[chunk_id].score = score
        return sorted(records.values(), key=lambda item: item.score, reverse=True)[:limit]
