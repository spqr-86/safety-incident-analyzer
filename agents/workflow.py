from typing import Dict, List, TypedDict

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph

from .relevance_checker import RelevanceChecker
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent

MAX_RERUNS = 1
TOPK_DOCS_FOR_AGENTS = 20  # Увеличили для глубокого анализа больших документов


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    research_thought: str  # Added field for thought chain
    verification_report: str
    is_relevant: bool
    retriever: BaseRetriever
    loops: int


class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        wf = StateGraph(AgentState)
        wf.add_node("check_relevance", self._check_relevance_step)
        wf.add_node("research", self._research_step)
        wf.add_node("verify", self._verification_step)
        wf.set_entry_point("check_relevance")
        wf.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END},
        )
        wf.add_edge("research", "verify")
        wf.add_conditional_edges(
            "verify", self._decide_next_step, {"re_research": "research", "end": END}
        )
        return wf.compile()

    def full_pipeline(self, question: str, retriever: BaseRetriever):
        docs = retriever.invoke(question)
        initial_state: AgentState = {
            "question": question,
            "documents": docs[:TOPK_DOCS_FOR_AGENTS],
            "draft_answer": "",
            "research_thought": "",
            "verification_report": "",
            "is_relevant": False,
            "retriever": retriever,
            "loops": 0,
        }
        final_state = self.compiled_workflow.invoke(initial_state)
        return {
            "draft_answer": final_state["draft_answer"],
            "research_thought": final_state.get("research_thought", ""),
            "verification_report": final_state["verification_report"],
        }

    def _check_relevance_step(self, state: AgentState) -> Dict:
        label = self.relevance_checker.check(
            question=state["question"], retriever=state["retriever"], k=20
        )
        if label in {"CAN_ANSWER", "PARTIAL"}:
            return {"is_relevant": True}
        return {
            "is_relevant": False,
            "draft_answer": "Недостаточно данных в загруженных документах для ответа на этот вопрос.",
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def _research_step(self, state: AgentState) -> Dict:
        res = self.researcher.generate(state["question"], state["documents"])
        return {
            "draft_answer": res["draft_answer"],
            "research_thought": res.get("thought", ""),
        }

    def _verification_step(self, state: AgentState) -> Dict:
        res = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": res["verification_report"]}

    def _decide_next_step(self, state: AgentState) -> str:
        report = state["verification_report"]
        needs_rerun = ("**Supported:** NO" in report) or ("**Relevant:** NO" in report)
        if needs_rerun and state["loops"] < MAX_RERUNS:
            # в идеале — сузить вопрос: извлечь из отчёта проблемные утверждения и уточнить промпт
            state["loops"] += 1
            return "re_research"
        return "end"
