"""FastAPI application factory and REST endpoints."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings
from .exceptions import ConfigurationError, DocumentValidationError, IndexNotReadyError
from .service import RAGService


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: dict[str, Any] | None = None


class CitationResponse(BaseModel):
    index: int
    chunk_id: str
    source: str
    page: int | None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    degraded: bool
    question: str
    rewritten_question: str
    latency_ms: float


class DocumentResponse(BaseModel):
    id: str
    name: str
    extension: str
    size: int
    sha256: str
    created_at: str
    chunk_count: int


class UploadDocumentResponse(DocumentResponse):
    created: bool


class SearchHitResponse(BaseModel):
    chunk_id: str
    source: str
    page: int | None
    score: float
    vector_score: float | None
    bm25_score: float | None
    rerank_score: float | None
    snippet: str


class RebuildResponse(BaseModel):
    documents: int
    chunks: int


def _service(request: Request) -> RAGService:
    return request.app.state.service


def _error_status(exc: Exception) -> int:
    if isinstance(exc, DocumentValidationError):
        return 400
    if isinstance(exc, IndexNotReadyError):
        return 409
    if isinstance(exc, ConfigurationError):
        return 503
    return 500


def create_app(
    service: RAGService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    app = FastAPI(
        title="RAG Knowledge Base API",
        version=__version__,
        description="Hybrid retrieval and citation-aware question answering.",
    )
    app.state.service = service or RAGService(resolved_settings)

    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = resolved_settings.api_key
        if expected and request.url.path != "/health":
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided, expected):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-API-Key"},
                )
        return await call_next(request)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return await run_in_threadpool(_service(request).health)

    @app.get("/api/v1/documents", response_model=list[DocumentResponse])
    async def list_documents(request: Request) -> list[DocumentResponse]:
        documents = await run_in_threadpool(_service(request).list_documents)
        return [DocumentResponse(**item.to_dict()) for item in documents]

    @app.post("/api/v1/documents", response_model=UploadDocumentResponse, status_code=201)
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
    ) -> UploadDocumentResponse:
        raw = await file.read(resolved_settings.max_upload_mb * 1024 * 1024 + 1)
        try:
            document, created = await run_in_threadpool(
                _service(request).ingest_bytes,
                file.filename or "upload",
                raw,
            )
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return UploadDocumentResponse(**document.to_dict(), created=created)

    @app.delete("/api/v1/documents/{doc_id}", status_code=204)
    async def delete_document(doc_id: str, request: Request) -> None:
        deleted = await run_in_threadpool(_service(request).delete_document, doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")

    @app.post("/api/v1/index/rebuild", response_model=RebuildResponse)
    async def rebuild_index(request: Request) -> RebuildResponse:
        try:
            payload = await run_in_threadpool(_service(request).rebuild_index)
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return RebuildResponse(**payload)

    @app.post("/api/v1/search", response_model=list[SearchHitResponse])
    async def search(payload: QueryRequest, request: Request) -> list[SearchHitResponse]:
        try:
            hits = await run_in_threadpool(
                _service(request).search,
                payload.question,
                top_k=payload.top_k,
                filters=payload.filters,
            )
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return [
            SearchHitResponse(
                chunk_id=hit.chunk.id,
                source=str(hit.chunk.metadata.get("source") or "unknown"),
                page=hit.chunk.metadata.get("page"),
                score=hit.score,
                vector_score=hit.vector_score,
                bm25_score=hit.bm25_score,
                rerank_score=hit.rerank_score,
                snippet=hit.chunk.text[:500],
            )
            for hit in hits
        ]

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query(payload: QueryRequest, request: Request) -> QueryResponse:
        try:
            result = await run_in_threadpool(
                _service(request).ask,
                payload.question,
                session_id=payload.session_id,
                top_k=payload.top_k,
                filters=payload.filters,
            )
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return QueryResponse(**result.to_dict())

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request) -> list[dict[str, str]]:
        return await run_in_threadpool(_service(request).sessions.get, session_id, 50)

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def clear_session(session_id: str, request: Request) -> None:
        await run_in_threadpool(_service(request).clear_session, session_id)

    return app
