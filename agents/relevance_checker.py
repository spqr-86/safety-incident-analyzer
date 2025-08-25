from typing import Dict, List

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from config import settings
from src.llm_factory import get_llm


class RelevanceChecker:
    """
    Двухступенчатая проверка:
    1) Heuristic: по ретриву — если мало релевантных чанков,
    считаем нерелевантным.
    2) (Опционально) LLM-критик — уточняет между PARTIAL/CAN_ANSWER.
    """

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self.llm = get_llm() if use_llm_judge else None

    def _heuristic(
        self,
        question: str,
        retriever: BaseRetriever,
        k: int = 20,
    ) -> Dict:
        docs: List[Document] = retriever.invoke(question)
        k = min(k, len(docs))
        if k == 0:
            return {"label": "NO_MATCH", "docs": []}
        # простая метрика: средняя длина/плотность и k
        topk = docs[:k]
        avg_len = sum(len(d.page_content) for d in topk) / k
        # эвристические пороги под СНиП/ГОСТ (длинный и насыщенный текст)
        if k >= 5 and avg_len > 300:
            label = "CAN_ANSWER"
        elif k >= 3 and avg_len > 150:
            label = "PARTIAL"
        else:
            label = "NO_MATCH"
        return {"label": label, "docs": topk}

    def _llm_judge(self, question: str, docs: List[Document], k: int = 6) -> str:
        if not self.llm:
            return "PARTIAL"
        passages = "\n\n".join(d.page_content for d in docs[:k])
        prompt = f"""
Ты — классификатор релевантности вопроса и текстовых фрагментов из 
нормативной базы (СНиП, ГОСТ, ОТ, ПБ).
Оцени, можно ли на основе фрагментов полностью ответить на вопрос.

Верни ровно один лейбл:
- CAN_ANSWER — можно полностью ответить;
- PARTIAL — тема есть, но ответа не хватает;
- NO_MATCH — тема не затрагивается.

Вопрос: {question}

Фрагменты:
{passages}

Ответи одним словом: CAN_ANSWER или PARTIAL или NO_MATCH
"""
        label = self.llm.invoke(prompt).content.strip().upper()
        return label if label in {"CAN_ANSWER", "PARTIAL", "NO_MATCH"} else "PARTIAL"

    def check(self, question: str, retriever: BaseRetriever, k=3) -> str:
        h = self._heuristic(question, retriever, k=k)
        if h["label"] == "NO_MATCH" or not self.use_llm_judge:
            return h["label"]
        # подтюним решением LLM между CAN_ANSWER / PARTIAL
        llm_label = self._llm_judge(question, h["docs"], k=6)
        # если эвристика сказала CAN_ANSWER, а LLM — NO_MATCH,
        # оставим PARTIAL (консервативно)
        if h["label"] == "CAN_ANSWER" and llm_label == "NO_MATCH":
            return "PARTIAL"
        return llm_label
