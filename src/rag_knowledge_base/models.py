"""Shared domain models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentMeta:
    id: str
    name: str
    stored_name: str
    extension: str
    size: int
    sha256: str
    created_at: str
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentMeta":
        return cls(**value)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(slots=True)
class LoadedPage:
    text: str
    page: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class Citation:
    index: int
    chunk_id: str
    source: str
    page: int | None
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    degraded: bool = False
    question: str = ""
    rewritten_question: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "degraded": self.degraded,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "latency_ms": self.latency_ms,
        }
