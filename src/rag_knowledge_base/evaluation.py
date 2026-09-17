"""Retrieval evaluation utilities for reproducible RAG experiments."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import RetrievedChunk


@dataclass(slots=True)
class EvaluationSummary:
    questions: int
    k: int
    hit_rate: float
    recall: float
    mrr: float
    precision: float
    avg_retrieval_latency_ms: float
    answer_keyword_coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation dataset must contain at least one question")
    required = {"id", "question", "relevant_sources"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Dataset row {index} is missing fields: {sorted(missing)}")
    return rows


def is_relevant(hit: RetrievedChunk, row: dict[str, Any]) -> bool:
    sources = {str(value) for value in row.get("relevant_sources", [])}
    source = str(hit.chunk.metadata.get("source", ""))
    if source not in sources:
        return False
    expected_pages = row.get("relevant_pages")
    if expected_pages:
        page = hit.chunk.metadata.get("page")
        if page is None or int(page) not in {int(value) for value in expected_pages}:
            return False
    expected_terms = [str(value).lower() for value in row.get("relevant_terms", [])]
    if expected_terms and not all(term in hit.chunk.text.lower() for term in expected_terms):
        return False
    return True


def _first_relevant_rank(hits: Iterable[RetrievedChunk], row: dict[str, Any]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, row):
            return rank
    return None


def evaluate_retrieval(
    dataset: list[dict[str, Any]],
    search_fn: Callable[[str, int], list[RetrievedChunk]],
    *,
    k: int = 5,
) -> EvaluationSummary:
    hit_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    precision_values: list[float] = []
    latencies: list[float] = []

    for row in dataset:
        started = time.perf_counter()
        hits = search_fn(str(row["question"]), k)
        latencies.append((time.perf_counter() - started) * 1000)
        relevant_hits = [hit for hit in hits if is_relevant(hit, row)]
        expected_sources = {str(value) for value in row.get("relevant_sources", [])}
        found_sources = {
            str(hit.chunk.metadata.get("source", "")) for hit in relevant_hits
        }
        hit_values.append(1.0 if relevant_hits else 0.0)
        recall_values.append(
            len(found_sources & expected_sources) / len(expected_sources)
            if expected_sources
            else 0.0
        )
        precision_values.append(len(relevant_hits) / len(hits) if hits else 0.0)
        rank = _first_relevant_rank(hits, row)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return EvaluationSummary(
        questions=len(dataset),
        k=k,
        hit_rate=statistics.fmean(hit_values),
        recall=statistics.fmean(recall_values),
        mrr=statistics.fmean(reciprocal_ranks),
        precision=statistics.fmean(precision_values),
        avg_retrieval_latency_ms=statistics.fmean(latencies),
    )


def evaluate_answers(
    dataset: list[dict[str, Any]],
    answer_fn: Callable[[str, int], str],
    *,
    k: int = 5,
) -> float:
    scores: list[float] = []
    for row in dataset:
        keywords = [str(value).lower() for value in row.get("expected_keywords", [])]
        if not keywords:
            continue
        answer = answer_fn(str(row["question"]), k).lower()
        scores.append(sum(keyword in answer for keyword in keywords) / len(keywords))
    if not scores:
        raise ValueError("Dataset has no expected_keywords for answer evaluation")
    return statistics.fmean(scores)
