import unittest

from rag_knowledge_base.config import Settings
from rag_knowledge_base.exceptions import ConfigurationError


class SettingsTests(unittest.TestCase):
    def test_explicit_empty_environment_uses_defaults(self):
        settings = Settings.from_env({}, load_file=False)
        self.assertEqual(settings.retrieval_mode, "hybrid")
        self.assertEqual(settings.chunk_size, 800)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_env(
                {"RKB_CHUNK_SIZE": "500", "RKB_CHUNK_OVERLAP": "500"},
                load_file=False,
            )


if __name__ == "__main__":
    unittest.main()
