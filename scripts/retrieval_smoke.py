"""Run a real embedding and retrieval smoke test against the sample document."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_knowledge_base.config import Settings  # noqa: E402
from rag_knowledge_base.evaluation import evaluate_retrieval, load_dataset  # noqa: E402
from rag_knowledge_base.service import RAGService  # noqa: E402


class OfflineChat:
    def generate(self, messages):
        raise RuntimeError("offline")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--question", default="差旅报销需要哪些材料？")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        settings = Settings.from_env(load_file=False)
        settings.data_dir = Path(directory)
        settings.device = args.device
        settings.retrieval_mode = "hybrid"
        if args.embedding_model:
            settings.embedding_model = args.embedding_model
        settings.ensure_directories()

        service = RAGService(settings, chat_factory=OfflineChat)
        document, created = service.ingest_file(
            ROOT / "examples" / "documents" / "sample-handbook.md"
        )
        hits = service.search(args.question, top_k=3)
        result = service.ask(args.question, top_k=3)
        dataset = load_dataset(ROOT / "examples" / "evaluation" / "questions.jsonl")
        summary = evaluate_retrieval(
            dataset,
            lambda question, top_k: service.search(question, top_k=top_k),
            k=3,
        )

        print(f"document={document.name} created={created} chunks={document.chunk_count}")
        print(
            f"eval hit_rate={summary.hit_rate:.3f} recall={summary.recall:.3f} "
            f"mrr={summary.mrr:.3f}"
        )
        for index, hit in enumerate(hits, start=1):
            print(
                f"hit={index} score={hit.score:.6f} source={hit.chunk.metadata['source']} "
                f"text={hit.chunk.text[:60]}"
            )
        print(f"fallback={result.degraded} answer={result.answer[:100]}")

        if not hits:
            raise SystemExit("Smoke test failed: retrieval returned no results")
        if "sample-handbook.md" not in {hit.chunk.metadata["source"] for hit in hits}:
            raise SystemExit("Smoke test failed: expected source was not retrieved")
        if not result.degraded:
            raise SystemExit("Smoke test failed: offline chat should trigger fallback")
        if summary.hit_rate < 0.8:
            raise SystemExit("Smoke test failed: retrieval hit rate is below 0.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
