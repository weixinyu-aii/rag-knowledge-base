PYTHON ?= python

.PHONY: install install-dev test lint compile api ui ingest evaluate clean

install:
	$(PYTHON) -m pip install -e ".[rag,api,ui]"

install-dev:
	$(PYTHON) -m pip install -e ".[api,dev]"

test:
	$(PYTHON) scripts/run_tests.py

lint:
	ruff check src tests app

compile:
	PYTHONPATH=src $(PYTHON) -m compileall -q src app

api:
	PYTHONPATH=src uvicorn rag_knowledge_base.api:create_app --factory --host 0.0.0.0 --port 8000 --reload

ui:
	PYTHONPATH=src $(PYTHON) -m rag_knowledge_base ui

ingest:
	PYTHONPATH=src $(PYTHON) -m rag_knowledge_base ingest examples/documents/sample-handbook.md

evaluate:
	PYTHONPATH=src $(PYTHON) -m rag_knowledge_base evaluate --dataset examples/evaluation/questions.jsonl --k 3

smoke:
	$(PYTHON) scripts/retrieval_smoke.py

check:
	$(PYTHON) scripts/check_project.py

clean:
	$(PYTHON) -m compileall -q -f src app
