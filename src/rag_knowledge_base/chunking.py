"""Deterministic, dependency-free text chunking."""

from __future__ import annotations

import re

from .models import Chunk, LoadedPage


_SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;]|$)", re.DOTALL)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if not compact:
        return []
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(compact) if match.group(0).strip()]


def _semantic_overlap(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    sentences = split_sentences(text)
    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if selected and total + len(sentence) > overlap:
            break
        if not selected and len(sentence) > overlap:
            return ""
        selected.append(sentence)
        total += len(sentence)
        if total >= overlap:
            break
    return " ".join(reversed(selected)).strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        value = current.strip()
        if value:
            chunks.append(value)
        current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            flush()
            step = chunk_size - overlap
            for start in range(0, len(sentence), step):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(sentence):
                    break
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        flush()
        available_overlap = max(0, chunk_size - len(sentence) - 1)
        effective_overlap = min(overlap, available_overlap)
        overlap_text = (
            _semantic_overlap(chunks[-1], effective_overlap)
            if effective_overlap and chunks
            else ""
        )
        current = f"{overlap_text} {sentence}".strip() if overlap_text else sentence

    flush()
    return chunks


def chunk_pages(
    pages: list[LoadedPage],
    *,
    doc_id: str,
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            metadata = {
                **page.metadata,
                "doc_id": doc_id,
                "source": source,
                "page": page.page,
                "chunk_index": chunk_index,
            }
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{chunk_index}",
                    text=text,
                    metadata=metadata,
                )
            )
            chunk_index += 1
    return chunks
