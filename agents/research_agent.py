from typing import Dict, List

from langchain.schema import Document

from src.llm_factory import get_llm
from src.prompt_manager import PromptManager


class ResearchAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt_manager = PromptManager()

    def generate(self, question: str, documents: List[Document]) -> Dict:
        prompt = self.prompt_manager.render(
            "research_agent", question=question, documents=documents
        )
        out = self.llm.invoke(prompt)
        raw = str(out.content).strip()

        # Попытка извлечь блоки <thought> и <answer>
        import re

        # Извлекаем с поддержкой любого регистра и многострочности
        thought_match = re.search(
            r"<thought>(.*?)</thought>", raw, re.DOTALL | re.IGNORECASE
        )
        answer_match = re.search(
            r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE
        )

        thought = thought_match.group(1).strip() if thought_match else ""
        answer = answer_match.group(1).strip() if answer_match else raw

        # Дополнительная очистка, если теги остались в ответе (например, если answer_match не сработал)
        if not answer_match:
            # Удаляем мысли из ответа, если они там остались
            answer = re.sub(
                r"<thought>.*?</thought>", "", answer, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            # Удаляем сами теги
            answer = re.sub(
                r"<(?:/)?(?:answer|thought)>", "", answer, flags=re.IGNORECASE
            ).strip()

        return {"draft_answer": answer, "thought": thought}
