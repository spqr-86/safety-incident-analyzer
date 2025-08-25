from typing import Dict, List

from langchain.schema import Document

from src.llm_factory import get_llm


class ResearchAgent:
    def __init__(self):
        self.llm = get_llm()

    def _prompt(self, question: str, docs: List[Document]) -> str:
        # соберём компактный контекст (можно обрезать по символам, если нужно)
        context = "\n\n".join(d.page_content for d in docs)
        # поощряем ссылки на пункты/разделы
        return f"""
Ты — ассистент по нормативной документации (СНиП, ГОСТ, ОТ, ПБ).
Отвечай ТОЛЬКО по контексту ниже. Где возможно, указывай конкретные ссылки:
"Источник: <source>, раздел/пункт: <...>".

Если информации недостаточно, честно напиши: "Недостаточно данных в предоставленных документах."

Вопрос: {question}

Контекст:
{context}

Формат ответа:
- Краткий ответ (2–5 предложений).
- Ссылки на документы (по возможности): Источник, номер документа, раздел/пункт/таблица.
"""

    def generate(self, question: str, documents: List[Document]) -> Dict:
        prompt = self._prompt(question, documents)
        out = self.llm.invoke(prompt)
        draft = out.content.strip()
        return {"draft_answer": draft}
