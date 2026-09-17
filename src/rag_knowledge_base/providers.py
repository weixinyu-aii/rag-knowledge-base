"""External model providers with narrow, testable interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .exceptions import ConfigurationError


def resolve_model_source(model_name: str, *, allow_download: bool) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if allow_download:
        return model_name
    raise ConfigurationError(
        f"本地模型不存在: {candidate}。已禁止联网下载；请设置正确的 "
        "RKB_EMBEDDING_MODEL/RKB_RERANKER_MODEL，或显式设置 "
        "RKB_ALLOW_MODEL_DOWNLOAD=true。"
    )


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class ChatProvider(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...


class SentenceTransformerEmbeddings:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        allow_download: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = resolve_model_source(
            model_name, allow_download=allow_download
        )
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            local_files_only=not allow_download,
        )
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_getter())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0]
        return vector.tolist()


class TongyiChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        from dashscope import Generation

        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            result_format="message",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if response.status_code != 200:
            message = getattr(response, "message", "unknown provider error")
            raise RuntimeError(f"DashScope request failed: {message}")
        choices = response.output.get("choices") or []
        if not choices:
            raise RuntimeError("DashScope returned no choices")
        return str(choices[0]["message"]["content"]).strip()
