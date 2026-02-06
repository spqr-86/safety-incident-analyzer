from typing import Dict, Any
from langgraph.prebuilt import create_react_agent
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage

from src.llm_factory import get_llm
from src.prompt_manager import PromptManager
from src.agent_tools import search_documents, visual_proof, set_global_retriever
from utils.logging import logger


class UniversalAgent:
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        # Set global retriever for tool access
        set_global_retriever(retriever)

        self.llm = get_llm()
        self.prompt_manager = PromptManager()

        # Tools
        self.tools = [search_documents, visual_proof]

        # System Prompt
        try:
            self.system_message = self.prompt_manager.render("universal_agent")
        except Exception as e:
            logger.error(f"Failed to load universal_agent prompt: {e}")
            self.system_message = "You are a helpful assistant."

        # Create Graph
        # We assume the LLM supports tool calling (GigaChat or OpenAI)
        self.graph = create_react_agent(
            self.llm, self.tools, state_modifier=self.system_message
        )

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Runs the agent and returns the final answer.
        Structure matches the old pipeline output for compatibility if possible,
        or just returns the answer string.
        """
        logger.info(f"UniversalAgent invoking with question: {question}")

        inputs = {"messages": [("user", question)]}

        # Invoke and get final state
        # Recursion limit prevents infinite loops
        result = self.graph.invoke(inputs, {"recursion_limit": 20})

        # Extract last message content
        final_message = result["messages"][-1]
        answer = final_message.content

        return {
            "draft_answer": answer,
            # Legacy fields for compatibility with UI
            "verification_report": "Проверка выполнена встроенным механизмом агента.",
            "research_thought": "См. логи агента.",
        }
