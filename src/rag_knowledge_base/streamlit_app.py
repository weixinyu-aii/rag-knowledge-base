"""Streamlit interface. Run with: python -m rag_knowledge_base ui."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_knowledge_base.config import Settings
from rag_knowledge_base.logging_config import configure_logging
from rag_knowledge_base.models import AnswerResult, Citation
from rag_knowledge_base.service import RAGService


class RemoteSessions:
    def __init__(self, backend: "APIBackend"):
        self.backend = backend

    def get(self, session_id: str, limit: int = 50) -> list[dict[str, str]]:
        payload = self.backend.request("GET", f"/api/v1/sessions/{quote(session_id)}")
        return list(payload)[-limit:]


class APIBackend:
    """Small Streamlit adapter that keeps UI and API processes stateless."""

    def __init__(self, base_url: str, api_key: str | None = None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(timeout=120)
        self.sessions = RemoteSessions(self)

    def request(self, method: str, path: str, **kwargs):
        response = self.client.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            **kwargs,
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"API {response.status_code}: {detail}")
        if response.status_code == 204:
            return None
        return response.json()

    def health(self) -> dict:
        return self.request("GET", "/health")

    def list_documents(self):
        return [SimpleNamespace(**item) for item in self.request("GET", "/api/v1/documents")]

    def ingest_bytes(self, filename: str, content: bytes):
        payload = self.request(
            "POST",
            "/api/v1/documents",
            files={"file": (filename, content, "application/octet-stream")},
        )
        return SimpleNamespace(**payload), bool(payload.get("created", True))

    def delete_document(self, doc_id: str) -> bool:
        try:
            self.request("DELETE", f"/api/v1/documents/{quote(doc_id)}")
            return True
        except RuntimeError as exc:
            if "API 404" in str(exc):
                return False
            raise

    def rebuild_index(self) -> dict[str, int]:
        return self.request("POST", "/api/v1/index/rebuild")

    def ask(
        self,
        question: str,
        *,
        session_id: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> AnswerResult:
        payload = self.request(
            "POST",
            "/api/v1/query",
            json={
                "question": question,
                "session_id": session_id,
                "top_k": top_k,
                "filters": filters,
            },
        )
        return AnswerResult(
            answer=payload["answer"],
            citations=[Citation(**item) for item in payload["citations"]],
            degraded=payload["degraded"],
            question=payload["question"],
            rewritten_question=payload["rewritten_question"],
            latency_ms=payload["latency_ms"],
        )

    def clear_session(self, session_id: str) -> None:
        self.request("DELETE", f"/api/v1/sessions/{quote(session_id)}")


st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_backend():
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    if settings.api_url:
        return APIBackend(settings.api_url, settings.api_key), settings
    return RAGService(settings), settings


backend, settings = get_backend()
health = backend.health()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("📚 RAG 企业知识库")
st.caption("BGE + FAISS + BM25 + RRF｜答案带引用｜支持 PDF / DOCX / Markdown / TXT / CSV")

with st.sidebar:
    st.header("知识库")
    uploads = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "md", "markdown", "txt", "csv"],
        accept_multiple_files=True,
    )
    if st.button("导入并建立索引", type="primary", width="stretch"):
        if not uploads:
            st.warning("请先选择文档")
        else:
            progress = st.progress(0, text="准备导入...")
            for index, uploaded in enumerate(uploads, start=1):
                progress.progress(
                    (index - 1) / len(uploads),
                    text=f"处理 {uploaded.name}...",
                )
                try:
                    document, created = backend.ingest_bytes(
                        uploaded.name, uploaded.getvalue()
                    )
                    action = "已导入" if created else "已存在，已跳过"
                    st.success(f"{action}: {document.name}，{document.chunk_count} 个文本块")
                except Exception as exc:
                    st.error(f"{uploaded.name} 导入失败: {exc}")
            progress.progress(1.0, text="完成")
            st.rerun()

    st.divider()
    documents = backend.list_documents()
    st.write(f"文档数：**{len(documents)}**｜文本块：**{health.get('chunks', 0)}**")
    if documents and st.button("重建全部索引", width="stretch"):
        with st.spinner("正在重建..."):
            result = backend.rebuild_index()
        st.success(f"完成：{result['documents']} 个文档，{result['chunks']} 个文本块")
        st.rerun()

    for document in documents:
        with st.expander(f"📄 {document.name}"):
            st.caption(f"ID: {document.id}")
            st.caption(f"类型：{document.extension}｜大小：{document.size / 1024:.1f} KB")
            st.caption(f"文本块：{document.chunk_count}")
            if st.button("删除", key=f"delete-{document.id}", width="stretch"):
                backend.delete_document(document.id)
                st.rerun()

    st.divider()
    st.caption(
        f"模型：{settings.llm_model}\n"
        f"嵌入：{settings.embedding_model}\n"
        f"检索：{settings.retrieval_mode}"
    )
    llm_configured = health.get("llm_configured", bool(settings.dashscope_api_key))
    if not llm_configured:
        st.warning("未配置 DASHSCOPE_API_KEY，问答将使用检索降级结果。")

if not health.get("index_ready"):
    st.info("知识库为空。请在左侧上传文档，或者先运行命令行导入。")
    st.stop()

left, right = st.columns([2, 1])
with right:
    top_k = st.slider("引用片段数量", min_value=1, max_value=10, value=settings.retriever_k)
    if st.button("清空当前对话", width="stretch"):
        backend.clear_session(st.session_state.session_id)
        st.rerun()

with left:
    question = st.chat_input("请输入问题，例如：差旅报销需要哪些材料？")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成答案..."):
                try:
                    result = backend.ask(
                        question,
                        session_id=st.session_state.session_id,
                        top_k=top_k,
                    )
                except Exception as exc:
                    st.error(f"请求失败：{exc}")
                    st.stop()
            st.write(result.answer)
            if result.degraded:
                st.warning("当前返回的是检索降级结果，未经过大模型生成。")
            with st.expander(f"引用来源（{len(result.citations)}）", expanded=True):
                for citation in result.citations:
                    page = f" · 第 {citation.page} 页" if citation.page else ""
                    st.markdown(f"**[{citation.index}] {citation.source}**{page}")
                    st.caption(citation.snippet[:220] + ("..." if len(citation.snippet) > 220 else ""))

with right:
    st.subheader("最近对话")
    history = backend.sessions.get(st.session_state.session_id, limit=8)
    if not history:
        st.caption("暂无对话")
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
