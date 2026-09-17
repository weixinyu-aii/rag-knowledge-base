# Contributing

## Development setup

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[api,dev]"
~~~

## Before opening a pull request

~~~bash
python scripts/run_tests.py
ruff check src tests app
python -m compileall -q src app
~~~

## Guidelines

- Keep the retrieval, provider, persistence, and API layers separated.
- Add or update tests for behavior changes.
- Do not commit API keys, uploaded documents, model weights, indexes, or databases.
- Explain retrieval or prompt changes together with evaluation impact when possible.
- Prefer small pull requests with a clear problem statement.
