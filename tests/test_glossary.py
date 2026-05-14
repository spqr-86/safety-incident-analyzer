import re
import unittest
from unittest.mock import patch

from src.glossary import expand_query_with_glossary, load_glossary


class TestGlossaryLoading(unittest.TestCase):
    def test_load_glossary_missing_file_returns_empty(self):
        self.assertEqual(load_glossary("/nonexistent/glossary.yaml"), {})

    def test_load_glossary_real_file_has_programma_b(self):
        glossary = load_glossary()
        self.assertIn("программа б", glossary)
        self.assertIn("безопасным методам", glossary["программа б"])


class TestExpandQuery(unittest.TestCase):
    @patch("src.glossary._compiled_patterns")
    def test_expand_query_with_match(self, mock_patterns):
        pattern = re.compile(r"программа\s+а\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "программа а", "программа обучения по общим вопросам ОТ")
        ]
        result = expand_query_with_glossary("Что такое программа А?")
        self.assertIn("[Глоссарий:", result)
        self.assertIn("программа обучения по общим вопросам ОТ", result)

    @patch("src.glossary._compiled_patterns")
    def test_expand_query_multiple_matches(self, mock_patterns):
        p1 = re.compile(r"программ\w*\s+а\b", re.IGNORECASE)
        p2 = re.compile(r"программ\w*\s+б\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (p1, "программа а", "программа по общим вопросам"),
            (p2, "программа б", "программа безопасных методов"),
        ]
        result = expand_query_with_glossary("Отличия программы А от программы Б?")
        self.assertIn("программа по общим вопросам", result)
        self.assertIn("программа безопасных методов", result)

    @patch("src.glossary._compiled_patterns")
    def test_expand_query_handles_declension(self, mock_patterns):
        pattern = re.compile(r"программ\w*\s+а\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "программа а", "программа обучения по общим вопросам")
        ]
        result = expand_query_with_glossary("Кто обучается по программе А?")
        self.assertIn("программа обучения по общим вопросам", result)

    @patch("src.glossary._compiled_patterns")
    def test_expand_query_empty_glossary_unchanged(self, mock_patterns):
        mock_patterns.return_value = []
        self.assertEqual(expand_query_with_glossary("Любой запрос"), "Любой запрос")

    @patch("src.glossary._compiled_patterns")
    def test_expand_query_no_match_unchanged(self, mock_patterns):
        pattern = re.compile(r"сиз\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "сиз", "средства индивидуальной защиты")
        ]
        self.assertEqual(
            expand_query_with_glossary("Запрос без терминов"), "Запрос без терминов"
        )

    @patch("src.glossary._compiled_patterns")
    def test_expand_query_case_insensitive(self, mock_patterns):
        pattern = re.compile(r"сиз\b", re.IGNORECASE)
        mock_patterns.return_value = [
            (pattern, "сиз", "средства индивидуальной защиты")
        ]
        result = expand_query_with_glossary("Требования к СИЗ на стройке")
        self.assertIn("средства индивидуальной защиты", result)

    def test_expand_query_real_glossary_programma_b(self):
        """The real glossary must bridge the short label 'программе Б' to the
        official term so retrieval can match the corpus chunk that uses the
        full name. This is the fix for the eval retrieval miss on 'программа Б'.
        """
        result = expand_query_with_glossary(
            "Какова минимальная продолжительность обучения по программе Б?"
        )
        self.assertNotEqual(
            result, "Какова минимальная продолжительность обучения по программе Б?"
        )
        self.assertIn("безопасным методам", result)

    def test_expand_query_no_false_positive_on_substring(self):
        """Abbreviations must match whole words only. The 4-letter term 'соут'
        must NOT fire on the unrelated word 'состоять' — that wrong expansion
        degrades retrieval.
        """
        q = "Из скольких человек должна состоять комиссия по проверке знаний?"
        self.assertEqual(expand_query_with_glossary(q), q)


if __name__ == "__main__":
    unittest.main()
