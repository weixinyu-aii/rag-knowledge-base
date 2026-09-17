import tempfile
import unittest
from pathlib import Path

from rag_knowledge_base.exceptions import ConfigurationError
from rag_knowledge_base.providers import resolve_model_source


class ModelPathTests(unittest.TestCase):
    def test_existing_local_model_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_model_source(directory, allow_download=False)
            self.assertEqual(Path(resolved), Path(directory).resolve())

    def test_missing_model_is_rejected_when_downloads_are_disabled(self):
        with self.assertRaises(ConfigurationError):
            resolve_model_source(
                "D:/not-existing/model",
                allow_download=False,
            )

    def test_missing_model_can_be_remote_only_when_explicitly_allowed(self):
        self.assertEqual(
            resolve_model_source("BAAI/test-model", allow_download=True),
            "BAAI/test-model",
        )


if __name__ == "__main__":
    unittest.main()
