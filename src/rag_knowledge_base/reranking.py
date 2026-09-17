"""Optional cross-encoder reranking."""

from __future__ import annotations

from typing import Sequence

from .models import RetrievedChunk
from .providers import resolve_model_source


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        allow_download: bool = False,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = resolve_model_source(
            model_name, allow_download=allow_download
        )
        self.model = CrossEncoder(
            self.model_name,
            device=device,
            local_files_only=not allow_download,
        )

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = self.model.predict(pairs)
        reranked: list[RetrievedChunk] = []
        for item, score in zip(candidates, scores, strict=True):
            item.rerank_score = float(score)
            item.score = float(score)
            reranked.append(item)
        reranked.sort(key=lambda value: value.rerank_score or float("-inf"), reverse=True)
        return reranked[:top_k]
