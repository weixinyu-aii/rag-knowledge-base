FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RKB_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md streamlit_app.py ./
COPY src ./src
COPY app ./app

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[rag,api,ui]"

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501


CMD ["uvicorn", "rag_knowledge_base.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
