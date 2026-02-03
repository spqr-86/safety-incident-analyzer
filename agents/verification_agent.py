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
        import json
        import re

        # 1. Сначала ищем в теге <json>...</json> (v2 формат)
        json_match = re.search(r"<json>(.*?)</json>", raw, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)
        else:
            # 2. Fallback: markdown code blocks (v1 формат или ошибка модели)
            code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if code_match:
                raw_json = code_match.group(1)
            else:
                # 3. Fallback: просто ищем {}
                brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
                raw_json = brace_match.group(1) if brace_match else raw

        try:
            data = json.loads(raw_json)
        except Exception:
            # Если совсем не распарсилось
            data = {
                "supported": "NO",
                "unsupported_claims": [],
                "contradictions": [],
                "relevant": "NO",
                "notes": f"Failed to parse verifier JSON. Raw output: {raw[:100]}...",
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
