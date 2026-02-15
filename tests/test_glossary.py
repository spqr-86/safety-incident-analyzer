import unittest
from unittest.mock import patch
from agents.multiagent_rag import _load_glossary, _expand_query


class TestGlossary(unittest.TestCase):
    """Test glossary loading and query expansion."""

    def test_load_glossary_from_file(self):
        _load_glossary()
        # This test depends on the real glossary file or lack thereof.
        # It's fine to keep it as basic verification.

    def test_load_glossary_missing_file(self):
        # We can test loading from a specific path if we use _load_glossary directly
        # but the cached version might interfere if we don't clear cache.
        # Also _load_glossary is cached based on path.
        # But here we are calling it with a new path.
        glossary = _load_glossary("/nonexistent/glossary.yaml")
        self.assertEqual(glossary, {})

    @patch("agents.multiagent_rag._compiled_glossary_patterns")
    def test_expand_query_with_match(self, mock_patterns):
        # Mock the patterns list directly.
        # Pattern structure: (regex_pattern, short_term, official_term)
        import re

        pattern = re.compile(r"программа\s+а\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "программа а", "программа обучения по общим вопросам ОТ")
        ]

        result = _expand_query("Что такое программа А?")
        self.assertIn("[Глоссарий:", result)
        self.assertIn("программа обучения по общим вопросам ОТ", result)

    @patch("agents.multiagent_rag._compiled_glossary_patterns")
    def test_expand_query_multiple_matches(self, mock_patterns):
        import re

        p1 = re.compile(r"программ\w*\s+а\b", re.IGNORECASE)
        p2 = re.compile(r"программ\w*\s+б\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (p1, "программа а", "программа по общим вопросам"),
            (p2, "программа б", "программа безопасных методов"),
        ]

        result = _expand_query("Отличия программы А от программы Б?")
        self.assertIn("программа по общим вопросам", result)
        self.assertIn("программа безопасных методов", result)

    @patch("agents.multiagent_rag._compiled_glossary_patterns")
    def test_expand_query_handles_declension(self, mock_patterns):
        import re

        # Simulating stem match for 'программы А'
        pattern = re.compile(r"программ\w*\s+а\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "программа а", "программа обучения по общим вопросам")
        ]

        result = _expand_query("Кто обучается по программе А?")
        self.assertIn("[Глоссарий:", result)
        self.assertIn("программа обучения по общим вопросам", result)

    @patch("agents.multiagent_rag._compiled_glossary_patterns")
    def test_expand_query_empty_glossary(self, mock_patterns):
        mock_patterns.return_value = []
        result = _expand_query("Любой запрос")
        self.assertEqual(result, "Любой запрос")

    @patch("agents.multiagent_rag._compiled_glossary_patterns")
    def test_expand_query_case_insensitive(self, mock_patterns):
        import re

        pattern = re.compile(r"сиз\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "сиз", "средства индивидуальной защиты")
        ]

        result = _expand_query("Требования к СИЗ на стройке")
        self.assertIn("средства индивидуальной защиты", result)


if __name__ == "__main__":
    unittest.main()
