import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.multiagent_rag import (
    MultiAgentRAGWorkflow,
    _load_glossary,
    _expand_query,
    _classify_query,
)
from src.parsers import (
    parse_status_block,
    parse_json_from_response,
)
from src.types import RAGStatus


class TestParseStatusBlock(unittest.TestCase):
    """Test the ===STATUS=== / ===ANSWER=== block parser."""

    def test_found_status(self):
        text = "===STATUS===\nFOUND\n===ANSWER===\nОтвет по охране труда."
        status, answer, unanswered = parse_status_block(text)
        self.assertEqual(status, RAGStatus.FOUND)
        self.assertEqual(answer, "Ответ по охране труда.")
        self.assertEqual(unanswered, [])

    def test_not_found_status(self):
        text = "===STATUS===\nNOT_FOUND\n===ANSWER===\nИнформация не найдена."
        status, answer, unanswered = parse_status_block(text)
        self.assertEqual(status, RAGStatus.NOT_FOUND)
        self.assertEqual(answer, "Информация не найдена.")

    def test_partial_with_unanswered(self):
        text = (
            "===STATUS===\nPARTIAL\n===ANSWER===\nЧастичный ответ.\n"
            "===UNANSWERED===\n- подвопрос 1\n- подвопрос 2\n"
        )
        status, answer, unanswered = parse_status_block(text)
        self.assertEqual(status, RAGStatus.PARTIAL)
        self.assertEqual(answer, "Частичный ответ.")
        self.assertEqual(unanswered, ["подвопрос 1", "подвопрос 2"])

    def test_fallback_no_markers(self):
        text = "Просто текст без маркеров."
        status, answer, unanswered = parse_status_block(text)
        self.assertEqual(status, RAGStatus.FOUND)  # default
        self.assertEqual(answer, "Просто текст без маркеров.")


class TestParseJsonFromResponse(unittest.TestCase):
    def test_raw_json(self):
        result = parse_json_from_response('{"type": "chitchat", "response": "Привет!"}')
        self.assertEqual(result["type"], "chitchat")

    def test_markdown_json(self):
        result = parse_json_from_response('```json\n{"type": "rag"}\n```')
        self.assertEqual(result["type"], "rag")

    def test_invalid_json(self):
        result = parse_json_from_response("not json at all")
        self.assertEqual(result, {})


class TestClassifyQuery(unittest.TestCase):
    """Test regex-based query classification."""

    def test_chitchat_greeting(self):
        self.assertEqual(_classify_query("Привет!"), "chitchat")
        self.assertEqual(_classify_query("Здравствуйте"), "chitchat")
        self.assertEqual(_classify_query("Добрый день!"), "chitchat")
        self.assertEqual(_classify_query("Спасибо!"), "chitchat")

    def test_chitchat_about_bot(self):
        self.assertEqual(_classify_query("Что ты умеешь?"), "chitchat")
        self.assertEqual(_classify_query("Кто ты?"), "chitchat")

    def test_out_of_scope(self):
        self.assertEqual(_classify_query("Какая погода?"), "out_of_scope")
        self.assertEqual(_classify_query("Расскажи анекдот"), "out_of_scope")
        self.assertEqual(_classify_query("Напиши стих"), "out_of_scope")

    def test_rag_query(self):
        self.assertEqual(_classify_query("Какой срок обучения по ОТ?"), "rag")
        self.assertEqual(_classify_query("Кто проводит инструктаж?"), "rag")
        self.assertEqual(_classify_query("Требования к СИЗ на стройке"), "rag")

    def test_rag_with_glossary_block(self):
        query = "Обучение по программе А?\n\n[Глоссарий: программа а → обучение по общим вопросам]"
        self.assertEqual(_classify_query(query), "rag")


class TestGlossary(unittest.TestCase):
    """Test glossary loading and query expansion."""

    def test_load_glossary_from_file(self):
        glossary = _load_glossary()
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


class TestMultiAgentWorkflow(unittest.TestCase):
    """Test the graph topology and routing logic."""

    def setUp(self):
        self.mock_retriever = MagicMock()

        # Patch LLM factories
        self.patcher_llm = patch("agents.multiagent_rag.get_llm")
        self.mock_get_llm = self.patcher_llm.start()
        self.mock_llm = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm

        # Initialize with OpenAI to avoid Gemini dependency
        self.workflow = MultiAgentRAGWorkflow(
            self.mock_retriever, llm_provider="openai"
        )

    def tearDown(self):
        self.patcher_llm.stop()

    def test_chitchat_direct_response(self):
        """Test filter → direct_response for chitchat."""
        result = self.workflow.invoke("Привет!")
        self.assertIn("Здравствуйте", result["final_answer"])

    def test_out_of_scope_direct_response(self):
        """Test filter → direct_response for out_of_scope."""
        result = self.workflow.invoke("Какая погода?")
        self.assertIn("за пределами моей компетенции", result["final_answer"])

    @patch("agents.multiagent_rag.create_react_agent")
    def test_rag_happy_path(self, mock_create_agent):
        """Test filter → rag_agent → verifier → format_final."""
        # Mock ReAct agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                HumanMessage(content="Какой срок?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_documents",
                            "args": {"query": "срок"},
                            "id": "1",
                        }
                    ],
                ),
                ToolMessage(
                    content="[Result 0] File: doc.pdf | Page: 5 | BBox: [0,0,100,100]\nExtended Context:\nСрок составляет 3 года.\n(IDs: [1])",
                    name="search_documents",
                    tool_call_id="1",
                ),
                AIMessage(
                    content="===STATUS===\nFOUND\n===ANSWER===\nСрок составляет 3 года. [Источник: doc.pdf, п. 5]"
                ),
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Verifier approves
        self.workflow.verifier_llm = MagicMock()
        self.workflow.verifier_llm.invoke.return_value = MagicMock(
            content='{"status": "approved", "issues": []}'
        )

        result = self.workflow.invoke("Какой срок обучения?")

        self.assertIn("Срок составляет 3 года", result["final_answer"])
        self.assertEqual(result["verify_status"], "approved")

    @patch("agents.multiagent_rag.create_react_agent")
    def test_verifier_revision_loop(self, mock_create_agent):
        """Test verifier needs_revision → re-run → approved."""
        # Agent returns FOUND both times
        agent_result = {
            "messages": [
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_documents",
                            "args": {"query": "инструктаж"},
                            "id": "1",
                        }
                    ],
                ),
                ToolMessage(
                    content="[Result 0] File: doc.pdf | Page: 3 | BBox: [0,0,100,100]\nExtended Context:\nИнструктаж раз в полгода.\n(IDs: [10])",
                    name="search_documents",
                    tool_call_id="1",
                ),
                AIMessage(
                    content="===STATUS===\nFOUND\n===ANSWER===\nИнструктаж проводится раз в полгода."
                ),
            ]
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = agent_result
        mock_create_agent.return_value = mock_agent

        # Verifier: first needs_revision, then approved
        self.workflow.verifier_llm = MagicMock()
        self.workflow.verifier_llm.invoke.side_effect = [
            MagicMock(
                content='{"status": "needs_revision", "issues": [{"type": "incomplete", "description": "Не указан пункт", "suggestion": "Добавь ссылку"}]}'
            ),
            MagicMock(content='{"status": "approved", "issues": []}'),
        ]

        result = self.workflow.invoke("Периодичность инструктажа?")

        # Agent called twice (initial + revision)
        self.assertEqual(mock_agent.invoke.call_count, 2)
        # Verifier called twice
        self.assertEqual(self.workflow.verifier_llm.invoke.call_count, 2)
        self.assertIn("Инструктаж", result["final_answer"])

    @patch("agents.multiagent_rag.create_react_agent")
    def test_max_revisions_stops_loop(self, mock_create_agent):
        """Test that revision loop stops after MAX_REVISIONS."""
        agent_result = {
            "messages": [
                HumanMessage(content="query"),
                AIMessage(content="===STATUS===\nFOUND\n===ANSWER===\nОтвет."),
            ]
        }
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = agent_result
        mock_create_agent.return_value = mock_agent

        # Verifier always needs_revision
        self.workflow.verifier_llm = MagicMock()
        self.workflow.verifier_llm.invoke.return_value = MagicMock(
            content='{"status": "needs_revision", "issues": [{"type": "incomplete", "description": "Неполно"}]}'
        )

        result = self.workflow.invoke("Вопрос?")

        # Flow: agent → verify(count=1, needs_revision) → agent(revision) → verify(count=2, needs_revision) → format_final
        self.assertEqual(mock_agent.invoke.call_count, 2)
        self.assertEqual(self.workflow.verifier_llm.invoke.call_count, 2)
        self.assertIn("⚠️", result["final_answer"])


class TestExtractStateFromMessages(unittest.TestCase):
    """Test the message history parser."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        with patch("agents.multiagent_rag.get_llm"):
            self.workflow = MultiAgentRAGWorkflow(
                self.mock_retriever, llm_provider="openai"
            )

    def test_extract_search_and_answer(self):
        messages = [
            HumanMessage(content="Вопрос"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "охрана труда"},
                        "id": "1",
                    }
                ],
            ),
            ToolMessage(
                content="[Result 0] File: test.pdf | Page: 1 | BBox: [0,0,100,100]\nExtended Context:\nТекст документа.\n(IDs: [1])",
                name="search_documents",
                tool_call_id="1",
            ),
            AIMessage(content="===STATUS===\nFOUND\n===ANSWER===\nОтвет найден."),
        ]

        result = self.workflow._extract_state_from_messages(messages)

        self.assertEqual(result["rag_status"], RAGStatus.FOUND)
        self.assertEqual(result["draft_answer"], "Ответ найден.")
        self.assertEqual(len(result["searches_performed"]), 1)
        self.assertEqual(result["searches_performed"][0]["query"], "охрана труда")
        self.assertEqual(len(result["chunks_found"]), 1)

    def test_extract_visual_proof_image_path(self):
        messages = [
            HumanMessage(content="q"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "visual_proof",
                        "args": {
                            "file_name": "f.pdf",
                            "page_no": 1,
                            "bbox": [0, 0, 100, 100],
                            "mode": "show",
                        },
                        "id": "2",
                    }
                ],
            ),
            ToolMessage(
                content="static/visuals/proof_abc123.png",
                name="visual_proof",
                tool_call_id="2",
            ),
            AIMessage(content="===STATUS===\nFOUND\n===ANSWER===\nОтвет."),
        ]

        result = self.workflow._extract_state_from_messages(messages)
        self.assertEqual(result["image_paths"], ["static/visuals/proof_abc123.png"])


if __name__ == "__main__":
    unittest.main()
