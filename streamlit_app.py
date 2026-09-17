"""Repository-level Streamlit entry point.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

MODULE_NAME = "rag_knowledge_base.streamlit_app"

# Streamlit re-executes this entry point on every interaction. Remove the
# package module from the import cache so the UI is rendered on every rerun.
sys.modules.pop(MODULE_NAME, None)
importlib.import_module(MODULE_NAME)
