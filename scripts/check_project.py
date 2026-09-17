"""Validate repository metadata and required files."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    ".gitignore",
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    "src/rag_knowledge_base/__init__.py",
    "tests/test_api.py",
]


def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        print("Missing required files: " + ", ".join(missing), file=sys.stderr)
        return 1
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    extras = set(project.get("optional-dependencies", {}))
    expected = {"rag", "api", "ui", "dev", "all"}
    if not expected.issubset(extras):
        print(f"Missing extras: {sorted(expected - extras)}", file=sys.stderr)
        return 1
    print(f"project={project['name']} version={project['version']} extras={sorted(extras)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
