"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from .exceptions import ConfigurationError


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer value: {value}") from exc


@dataclass(slots=True)
class Settings:
    dashscope_api_key: str | None = None
    llm_model: str = "qwen-turbo"
    embedding_model: str = "D:/langchain01/models/bge-small-zh-v1.5"
    device: str = "cpu"
    data_dir: Path = Path("./data")
    chunk_size: int = 800
    chunk_overlap: int = 150
    retriever_k: int = 5
    candidate_k: int = 20
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    reranker_enabled: bool = False
    reranker_model: str = "D:/langchain01/models/bge-reranker-base"
    max_upload_mb: int = 20
    api_key: str | None = None
    api_url: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:8501"])
    log_level: str = "INFO"
    allow_model_download: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_file: bool = True,
    ) -> "Settings":
        if load_file:
            load_dotenv()
        values = os.environ if env is None else env
        data_dir = Path(values.get("RKB_DATA_DIR", "./data")).expanduser().resolve()
        cors = [
            item.strip()
            for item in values.get("RKB_CORS_ORIGINS", "http://localhost:8501").split(",")
            if item.strip()
        ]
        settings = cls(
            dashscope_api_key=values.get("DASHSCOPE_API_KEY") or None,
            llm_model=values.get("RKB_LLM_MODEL", "qwen-turbo"),
            embedding_model=values.get(
                "RKB_EMBEDDING_MODEL", "D:/langchain01/models/bge-small-zh-v1.5"
            ),
            device=values.get("RKB_DEVICE", "cpu"),
            data_dir=data_dir,
            chunk_size=_as_int(values.get("RKB_CHUNK_SIZE"), 800),
            chunk_overlap=_as_int(values.get("RKB_CHUNK_OVERLAP"), 150),
            retriever_k=_as_int(values.get("RKB_RETRIEVER_K"), 5),
            candidate_k=_as_int(values.get("RKB_CANDIDATE_K"), 20),
            retrieval_mode=values.get("RKB_RETRIEVAL_MODE", "hybrid").lower(),
            rrf_k=_as_int(values.get("RKB_RRF_K"), 60),
            reranker_enabled=_as_bool(values.get("RKB_RERANKER_ENABLED"), False),
            reranker_model=values.get(
                "RKB_RERANKER_MODEL", "D:/langchain01/models/bge-reranker-base"
            ),
            max_upload_mb=_as_int(values.get("RKB_MAX_UPLOAD_MB"), 20),
            api_key=values.get("RKB_API_KEY") or None,
            api_url=values.get("RKB_API_URL") or None,
            cors_origins=cors,
            log_level=values.get("RKB_LOG_LEVEL", "INFO").upper(),
            allow_model_download=_as_bool(
                values.get("RKB_ALLOW_MODEL_DOWNLOAD"), False
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ConfigurationError("RKB_CHUNK_SIZE must be at least 100")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "RKB_CHUNK_OVERLAP must be non-negative and smaller than chunk size"
            )
        if self.retriever_k < 1 or self.candidate_k < self.retriever_k:
            raise ConfigurationError(
                "RKB_RETRIEVER_K must be positive and RKB_CANDIDATE_K >= it"
            )
        if self.retrieval_mode not in {"vector", "bm25", "hybrid"}:
            raise ConfigurationError(
                "RKB_RETRIEVAL_MODE must be vector, bm25, or hybrid"
            )
        if self.max_upload_mb < 1:
            raise ConfigurationError("RKB_MAX_UPLOAD_MB must be positive")

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def session_db_path(self) -> Path:
        return self.data_dir / "sessions.sqlite3"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
