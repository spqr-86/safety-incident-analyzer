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
        draft = str(out.content).strip()
        return {"draft_answer": draft}
