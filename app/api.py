"""Uvicorn entry point: uvicorn app.api:app --reload."""

from rag_knowledge_base.api import create_app

app = create_app()
