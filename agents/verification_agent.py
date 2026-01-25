from typing import Dict, List

from langchain.schema import Document

from src.llm_factory import get_llm


class VerificationAgent:
    def __init__(self):
        self.llm = get_llm()

    def _prompt(self, answer: str, docs: List[Document]) -> str:
        context = "\n\n".join(d.page_content for d in docs)
        return f"""
Ты — проверяющий по нормативной документации (СНиП, ГОСТ, ОТ, ПБ).
Проверь, что ответ поддержан контекстом и релевантен вопросу.

Верни строго JSON с ключами:
{{
  "supported": "YES|NO",
  "unsupported_claims": ["..."],
  "contradictions": ["..."],
  "relevant": "YES|NO",
  "notes": "строка"
}}

Ответ для проверки:
{answer}

Контекст:
{context}
"""

    def check(self, answer: str, documents: List[Document]) -> Dict:
        out = self.llm.invoke(self._prompt(answer, documents))
        raw = str(out.content).strip()
        # мягкий парсер JSON (LLM может иногда ошибиться)
        import json

        try:
            data = json.loads(raw)
        except Exception:
            data = {
                "supported": "NO",
                "unsupported_claims": [],
                "contradictions": [],
                "relevant": "NO",
                "notes": "Failed to parse verifier JSON.",
            }
        # человекочитаемый отчёт
        report = []
        report.append(f"**Supported:** {data.get('supported', 'NO')}")
        uc = data.get("unsupported_claims", []) or []
        report.append("**Unsupported Claims:** " + (", ".join(uc) if uc else "None"))
        ct = data.get("contradictions", []) or []
        report.append("**Contradictions:** " + (", ".join(ct) if ct else "None"))
        report.append(f"**Relevant:** {data.get('relevant', 'NO')}")
        notes = data.get("notes") or "None"
        report.append(f"**Additional Details:** {notes}")
        return {"verification_report": "\n".join(report)}
