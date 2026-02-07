from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from src.llm_factory import get_llm
from src.agent_tools import search_documents, visual_proof, set_global_retriever
from src.prompt_manager import PromptManager


class UniversalAgent:
    def __init__(self, retriever):
        self.llm = get_llm()
        self.retriever = retriever

        # Initialize tools with retriever context
        set_global_retriever(retriever)
        self.tools = [search_documents, visual_proof]

        # Load system prompt
        pm = PromptManager()
        # We assume universal_v1.j2 exists in prompts/agents/
        # Check registry or file directly.
        # Since PromptManager might expect a registry entry, we'll try to use it if registered,
        # otherwise we'll read the file directly if needed.
        # But let's assume it's registered as 'universal_agent'.
        # If not, we might need to add it to registry.yaml.
        try:
            self.system_prompt = pm.render("universal_agent")
        except Exception:
            # Fallback: read directly if not in registry
            with open("prompts/agents/universal_v1.j2", "r") as f:
                self.system_prompt = f.read()

        # Create ReAct Agent
        # prompt allows us to set the system prompt
        self.agent = create_react_agent(
            self.llm, self.tools, prompt=self.system_prompt, checkpointer=MemorySaver()
        )

    def invoke(self, question: str):
        """
        Run the agent with the given question.
        Returns a dict with 'draft_answer'.
        """
        inputs = {"messages": [("user", question)]}

        # Run the graph
        # config is needed for checkpointer (thread_id)
        config = {"configurable": {"thread_id": "default_thread"}}

        result = self.agent.invoke(inputs, config=config)

        # Extract the final message content
        last_message = result["messages"][-1]
        return {"draft_answer": last_message.content}
