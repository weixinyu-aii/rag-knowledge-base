import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "streamlit is not installed")
class StreamlitEntryTests(unittest.TestCase):
    def test_root_entry_renders_again_on_rerun(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"RKB_DATA_DIR": directory}, clear=False):
                app = AppTest.from_file(str(app_path), default_timeout=30).run()
                self.assertEqual(len(app.title), 1)
                self.assertFalse(app.exception)

                app.run()
                self.assertEqual(len(app.title), 1)
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
