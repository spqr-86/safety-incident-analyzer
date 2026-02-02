from typing import Dict, List

from langchain.schema import Document

from src.llm_factory import get_llm
from src.prompt_manager import PromptManager


class VerificationAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt_manager = PromptManager()

    def check(self, answer: str, documents: List[Document]) -> Dict:
        prompt = self.prompt_manager.render(
            "verification_agent", answer=answer, documents=documents
        )
        out = self.llm.invoke(prompt)
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
