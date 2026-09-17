"""Command line interface for ingestion, querying, evaluation, API, and UI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .evaluation import evaluate_answers, evaluate_retrieval, load_dataset
from .logging_config import configure_logging
from .service import RAGService


def _service() -> RAGService:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    return RAGService(settings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rkb",
        description="Hybrid RAG knowledge base toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest one or more documents")
    ingest.add_argument("files", nargs="+", help="Local file paths")

    ask = subparsers.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--session", default="default")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--json", action="store_true")

    subparsers.add_parser("list", help="List indexed documents")

    delete = subparsers.add_parser("delete", help="Delete a document")
    delete.add_argument("doc_id")

    subparsers.add_parser("rebuild", help="Rebuild all indexes from uploaded files")

    evaluate = subparsers.add_parser("evaluate", help="Run retrieval evaluation")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--with-answers", action="store_true")
    evaluate.add_argument("--output")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser("ui", help="Run the Streamlit interface")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "rag_knowledge_base.api:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0
    if args.command == "ui":
        script = Path(__file__).with_name("streamlit_app.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)])

    service = _service()
    if args.command == "ingest":
        for file_path in args.files:
            document, created = service.ingest_file(file_path)
            state = "created" if created else "already exists"
            print(f"{state}: {document.name} ({document.id}), chunks={document.chunk_count}")
        return 0

    if args.command == "ask":
        result = service.ask(args.question, session_id=args.session, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(result.answer)
            if result.citations:
                print("\nSources:")
                for citation in result.citations:
                    page = f", page {citation.page}" if citation.page else ""
                    print(f"[{citation.index}] {citation.source}{page}")
        return 0

    if args.command == "list":
        for document in service.list_documents():
            print(
                f"{document.id}\t{document.name}\t"
                f"{document.chunk_count} chunks\t{document.size} bytes"
            )
        return 0

    if args.command == "delete":
        if not service.delete_document(args.doc_id):
            print(f"Document not found: {args.doc_id}", file=sys.stderr)
            return 1
        print(f"Deleted: {args.doc_id}")
        return 0

    if args.command == "rebuild":
        print(json.dumps(service.rebuild_index(), ensure_ascii=False))
        return 0

    if args.command == "evaluate":
        dataset = load_dataset(args.dataset)
        summary = evaluate_retrieval(
            dataset,
            lambda question, top_k: service.search(question, top_k=top_k),
            k=args.k,
        )
        if args.with_answers:
            summary.answer_keyword_coverage = evaluate_answers(
                dataset,
                lambda question, top_k: service.ask(question, top_k=top_k).answer,
                k=args.k,
            )
        payload = summary.to_dict()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    return 1
