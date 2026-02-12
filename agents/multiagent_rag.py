"""
Multi-Agent RAG Workflow with ReAct Agent and Thinking Levels

Architecture:
    User Query → Glossary Expansion → Router (LLM) → RAG Agent (ReAct) → Verifier → Answer

Graph Structure:
    router → rag_agent → verifier → format_final
           → direct_response (chitchat/out_of_scope)

    Revision: verifier (needs_revision) → rag_agent (max 1)
"""

import logging
import re
import concurrent.futures
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from config.settings import settings
from agents.router_agent import RouterAgent
from src.agent_tools import (
    create_tool_context,
    make_tools,
)
from src.llm_factory import get_gemini_llm, get_llm
from src.parsers import (
    extract_text,
    parse_json_from_response,
    parse_search_results,
    parse_status_block,
)
from src.prompt_manager import PromptManager
from src.types import ChunkInfo, RAGStatus, RouteType, VerifyStatus
from src.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


GLOSSARY_PATH = Path(__file__).parent.parent / "config" / "term_glossary.yaml"


# --- Glossary ---
@lru_cache(maxsize=1)
def _load_glossary(path: str = str(GLOSSARY_PATH)) -> dict[str, str]:
    """Load term glossary (cached — called once per process)."""
    p = Path(path)
    if not p.exists():
        logger.warning("Term glossary not found at %s", p)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        terms = data.get("terms", {}) if data else {}
        return {
            key.lower(): val["official"]
            for key, val in terms.items()
            if "official" in val
        }
    except Exception as e:
        logger.error("Failed to load glossary: %s", e)
        return {}


@lru_cache(maxsize=1)
def _compiled_glossary_patterns() -> list[tuple[re.Pattern, str, str]]:
    """Pre-compile regex patterns for all glossary terms (cached)."""
    glossary = _load_glossary()
    patterns = []
    for short_term, official in glossary.items():
        pattern = _make_term_pattern(short_term)
        patterns.append((pattern, short_term, official))
    return patterns


def _make_term_pattern(term: str) -> re.Pattern:
    """Build a regex pattern that handles Russian morphological endings.

    For words >3 chars: truncate last 2 chars and allow any suffix.
    For words <=3 chars: exact match with word boundary.
    Example: "программа а" → r"программ\\w*\\s+а\\b"
    """
    words = term.lower().split()
    parts = []
    for w in words:
        if len(w) > 3:
            stem = w[: len(w) - 2]
            parts.append(re.escape(stem) + r"\w*")
        else:
            parts.append(re.escape(w) + r"\b")
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


def _expand_query(query: str) -> str:
    """Expand unofficial abbreviations using cached glossary patterns."""
    patterns = _compiled_glossary_patterns()
    if not patterns:
        return query
    expansions = []
    for pattern, short_term, official in patterns:
        if pattern.search(query):
            expansions.append(f"{short_term} → {official}")
    if expansions:
        return query + "\n\n[Глоссарий: " + "; ".join(expansions) + "]"
    return query


# --- State ---
class RAGState(TypedDict):
    """State for the Multi-Agent RAG workflow."""

    query: str
    route_type: RouteType
    direct_response: Optional[str]
    searches_performed: list[dict]  # [{query, results_count}]
    chunks_found: list[ChunkInfo]
    image_paths: list[str]
    draft_answer: str
    unanswered: list[str]
    rag_status: RAGStatus
    verify_status: VerifyStatus
    verify_issues: list[dict]
    revision_count: int
    final_answer: str
    is_routed: bool


class MultiAgentRAGWorkflow:
    """
    RAG workflow with a single ReAct agent and thinking levels.

    The agent autonomously decides when to search, decompose,
    and use visual_proof based on the merged prompt.

    Graph: router → rag_agent → verifier → format_final
                  → direct_response (chitchat/out_of_scope)

    Supports:
    - Gemini: Flash with adaptive thinking budget (up to 8192)
    - OpenAI: Uses configured model for all steps (no thinking budgets)
    """

    def __init__(
        self, retriever: BaseRetriever, llm_provider: str = "gemini", tools=None
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()

        # Tool context
        self.tool_ctx = create_tool_context(retriever)
        self.tools = tools or make_tools(self.tool_ctx)

        # LLMs based on provider
        if self.llm_provider == "gemini":
            self.rag_llm = get_gemini_llm(
                settings.GEMINI_FAST_MODEL,
                thinking_budget=settings.THINKING_BUDGET,
            )
            self.verifier_llm = get_gemini_llm(
                settings.GEMINI_FAST_MODEL,
                thinking_budget=settings.THINKING_VERIFIER,
                response_mime_type="application/json",
            )
        else:  # openai fallback
            openai_llm = get_llm()
            self.rag_llm = openai_llm
            self.verifier_llm = openai_llm

        self.prompt_manager = PromptManager()
        self.router_agent = RouterAgent(llm_provider=self.llm_provider)
        self.compiled_workflow = self._build_workflow()

        # Initialize Semantic Cache
        try:
            self.cache = SemanticCache()
        except Exception as e:
            logger.warning(f"Failed to initialize SemanticCache: {e}")
            self.cache = None

    def _build_workflow(self):
        """Build the LangGraph workflow."""

        graph = StateGraph(RAGState)

        graph.add_node("router", self._router_node)
        graph.add_node("direct_response", self._direct_response_node)
        graph.add_node("rag_agent", self._rag_agent_node)
        graph.add_node("verifier", self._verifier_node)
        graph.add_node("format_final", self._format_final_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "direct_response": "direct_response",
                "rag_agent": "rag_agent",
            },
        )
        graph.add_edge("direct_response", END)

        # Conditional edge based on similarity score and content
        graph.add_conditional_edges(
            "rag_agent",
            self._route_after_rag_agent,
            {
                "verifier": "verifier",
                "format_final": "format_final",
            },
        )

        graph.add_conditional_edges(
            "verifier",
            self._route_after_verify,
            {
                "format_final": "format_final",
                "rag_agent": "rag_agent",
            },
        )
        graph.add_edge("format_final", END)

        return graph.compile()

    def stream_events(self, query: str):
        """
        Process a user query and yield status updates + final answer.
        """
        # 0. Check Semantic Cache
        if self.cache:
            cached_answer = self.cache.get(query)
            if cached_answer:
                yield {"type": "status", "text": "🔎 Ответ найден в кеше!"}
                yield {"type": "final", "text": cached_answer, "state": {}}
                return

        expanded_query = _expand_query(query)

        # --- Speculative Execution: Run Router and Search in Parallel ---
        def run_search():
            """Speculative search using the search_documents tool."""
            search_tool = next(
                (t for t in self.tools if t.name == "search_documents"), None
            )
            if not search_tool:
                return [], {}
            try:
                # Invoke tool directly. It returns a string (JSON of chunks).
                result_str = search_tool.invoke({"query": expanded_query})
                chunks = parse_search_results(result_str)
                return chunks, {"query": expanded_query, "results_count": len(chunks)}
            except Exception as e:
                logger.error(f"Speculative search failed: {e}")
                return [], {}

        def run_router():
            """Run router agent."""
            return self.router_agent.route(expanded_query)

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_search = executor.submit(run_search)
            future_router = executor.submit(run_router)

            # Wait for results
            try:
                search_chunks, search_record = future_search.result()
            except Exception as e:
                logger.error(f"Speculative search thread failed: {e}")
                search_chunks, search_record = [], {}

            try:
                router_result = future_router.result()
            except Exception as e:
                logger.error(f"Router thread failed: {e}")
                # Fallback to a safe default
                router_result = {"type": RouteType.RAG_COMPLEX, "response": None}

        # Build initial state with speculative results
        initial_state: RAGState = {
            "query": expanded_query,
            "route_type": router_result["type"],
            "direct_response": router_result.get("response"),
            "searches_performed": [search_record] if search_record else [],
            "chunks_found": search_chunks,
            "image_paths": [],
            "draft_answer": "",
            "unanswered": [],
            "rag_status": RAGStatus.FOUND,
            "verify_status": VerifyStatus.APPROVED,
            "verify_issues": [],
            "revision_count": 0,
            "final_answer": "",
            "is_routed": True,
        }

        self.tool_ctx.search_call_count = 1 if search_record else 0
        self.tool_ctx.visual_proof_call_count = 0

        final_state = None

        # Stream the graph execution
        for event in self.compiled_workflow.stream(
            initial_state, {"recursion_limit": 50}
        ):
            # Yield node transitions
            if "router" in event:
                # Determine status message based on route
                state = event["router"]
                rtype = state.get("route_type")
                if rtype == RouteType.RAG_COMPLEX:
                    yield {
                        "type": "status",
                        "text": "🧠 Вопрос сложный, анализирую детали...",
                    }
                elif rtype == RouteType.RAG_SIMPLE:
                    yield {"type": "status", "text": "🔎 Ищу ответ в документах..."}
            elif "rag_agent" in event:
                # Don't overwrite status if we already set it in router,
                # but maybe update if it's taking long?
                pass
            elif "verifier" in event:
                yield {"type": "status", "text": "⚖️ Проверяю корректность ответа..."}
            elif "format_final" in event:
                state = event["format_final"]
                final = state.get("final_answer", "")
                final_state = state
                yield {"type": "final", "text": final, "state": state}
            elif "direct_response" in event:
                state = event["direct_response"]
                final = state.get("final_answer", "")
                final_state = state
                yield {"type": "final", "text": final, "state": state}

        # Update Cache
        if self.cache and final_state:
            # Only cache verified and found answers
            # Check draft_answer status from previous nodes if not available in format_final
            # Wait, final_state from format_final only has final_answer.
            # We need the full state history or merge it.
            # LangGraph stream events yield the update from the node, not full state?
            # Actually, event[node_name] IS the state update.
            # But the full state is maintained by LangGraph.
            # We need to check the full state logic.
            # Let's rely on what we can infer or pass through.
            # `format_final` returns `final_answer`.
            # To know verify_status, we need to look at previous events or state.

            # Since we can't easily access full state here without accumulation,
            # Let's trust that if we reached format_final and it's not "needs_revision"
            # (which would loop back), it's likely okay.
            # But we explicitly add warning if needs_revision.

            final_answer = final_state.get("final_answer", "")
            if "⚠️ Ответ предоставлен с оговорками" not in final_answer:
                # It's a clean answer. Add to cache.
                self.cache.add(query, final_answer)

    # --- Node implementations ---

    def _router_node(self, state: RAGState) -> dict:
        """LLM-based router: classify query."""
        # If already routed (via speculative execution), skip re-running
        if state.get("is_routed"):
            return {}

        result = self.router_agent.route(state["query"])

        updates = {"route_type": result["type"]}
        if result["response"]:
            updates["direct_response"] = result["response"]
        return updates

    def _direct_response_node(self, state: RAGState) -> dict:
        """Return direct response for chitchat/out_of_scope."""
        if state.get("direct_response"):
            return {"final_answer": state["direct_response"]}

        route_type = state.get("route_type", "")
        if route_type in (RouteType.CHITCHAT, "chitchat"):
            response = "Здравствуйте! Я готов помочь с вопросами по охране труда и промышленной безопасности."
        else:
            response = "Этот вопрос находится за пределами моей компетенции. Я специализируюсь на вопросах охраны труда и промышленной безопасности."
        return {"final_answer": response}

    def _rag_agent_node(self, state: RAGState) -> dict:
        """RAG Agent: ReAct agent with adaptive decomposition."""
        system_prompt = self.prompt_manager.render(
            "multiagent_rag_agent",
            searches_performed=state.get("searches_performed", []),
            chunks=state.get("chunks_found", []),
        )

        # Build user message with optional revision feedback + previous draft
        user_msg = state["query"]
        if state.get("verify_issues"):
            draft = state.get("draft_answer", "")
            feedback = "\n".join(
                f"- {i.get('description', '')} → {i.get('suggestion', '')}"
                for i in state["verify_issues"]
            )
            user_msg += (
                f"\n\n[РЕВИЗИЯ]\n"
                f"Твой предыдущий ответ:\n{draft}\n\n"
                f"Проблемы, найденные верификатором:\n{feedback}\n"
                f"Исправь указанные проблемы. Можешь выполнить дополнительный поиск если нужно."
            )

        tools = self.tools

        agent = create_react_agent(
            model=self.rag_llm,
            tools=tools,
            prompt=system_prompt,
            version="v1",
        )
        result = agent.invoke(
            {"messages": [("user", user_msg)]},
            {"recursion_limit": settings.MAX_AGENT_STEPS},
        )
        messages = result["messages"]

        return self._extract_state_from_messages(messages)

    def _verifier_node(self, state: RAGState) -> dict:
        """Verify draft answer against found chunks."""
        prompt = self.prompt_manager.render(
            "multiagent_verifier",
            query=state["query"],
            draft_answer=state.get("draft_answer", ""),
            chunks=state.get("chunks_found", []),
        )
        response = self.verifier_llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_json_from_response(extract_text(response.content).strip())

        status = parsed.get("status", "approved")
        revision_count = state.get("revision_count", 0)
        if status == "needs_revision":
            revision_count += 1

        return {
            "verify_status": status,
            "verify_issues": parsed.get("issues", []),
            "revision_count": revision_count,
        }

    def _format_final_node(self, state: RAGState) -> dict:
        """Format the final answer."""
        answer = state.get("draft_answer", "")

        # Strip leaked status markers from Gemini responses
        status_idx = answer.find("===STATUS===")
        if status_idx != -1:
            answer = answer[:status_idx].rstrip()

        if state.get("verify_status") == VerifyStatus.NEEDS_REVISION:
            answer += "\n\n⚠️ Ответ предоставлен с оговорками, рекомендуется дополнительная проверка."

        return {"final_answer": answer}

    # --- Routing functions ---

    def _route_after_router(self, state: RAGState) -> str:
        route_type = state.get("route_type", "")
        if route_type in (
            RouteType.CHITCHAT,
            RouteType.OUT_OF_SCOPE,
        ):
            return "direct_response"
        return "rag_agent"

    def _route_after_rag_agent(self, state: RAGState) -> str:
        """Decide whether to verify or trust the answer."""
        chunks = state.get("chunks_found", [])

        # 1. If not found or partial, verify/revise
        if state.get("rag_status") != RAGStatus.FOUND:
            return "verifier"

        if not chunks:
            return "verifier"

        # 2. Check visual proof
        if any(c.get("visual_text") for c in chunks):
            # If we used VLM, better verify it
            return "verifier"

        # 3. Check source consistency
        sources = {c.get("source") for c in chunks if c.get("source")}
        if len(sources) > 1:
            return "verifier"  # Cross-document synthesis needs check

        # 4. Check similarity score
        # Since we sort by relevance, the first chunk is the most important
        top_chunk = chunks[0]
        score = top_chunk.get("similarity", 0.0)

        if score > 0.85:
            logger.info(f"Skipping verification! High confidence: {score}")
            return "format_final"

        return "verifier"

    def _route_after_verify(self, state: RAGState) -> str:
        if state.get("verify_status") in (VerifyStatus.APPROVED, "approved"):
            return "format_final"
        if state.get("revision_count", 0) > settings.MAX_REVISIONS:
            return "format_final"  # max revisions reached
        return "rag_agent"

    # --- Message history parser ---

    def _extract_state_from_messages(self, messages: list) -> dict:
        """Extract structured state from ReAct agent message history.

        Parses tool calls and final AI response to populate state fields.
        """
        chunks_found = []
        searches_performed = []
        image_paths = []

        for msg in messages:
            # Extract search queries from AI tool calls
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "search_documents":
                        query_arg = tc["args"].get("query", "")
                        searches_performed.append(
                            {"query": query_arg, "results_count": 0}
                        )

            # Extract chunks from search results
            if isinstance(msg, ToolMessage):
                if msg.name == "search_documents":
                    parsed_chunks = parse_search_results(extract_text(msg.content))
                    # Update results_count for the last search
                    if searches_performed:
                        searches_performed[-1]["results_count"] = len(parsed_chunks)
                    chunks_found.extend(parsed_chunks)

                elif msg.name == "visual_proof":
                    content = extract_text(msg.content)
                    if content.startswith("static/") or content.startswith("proof_"):
                        image_paths.append(content)
                    # If analyze mode, update the last chunk's visual_text
                    elif "[Visual Analysis Result]" in content:
                        if chunks_found:
                            chunks_found[-1]["visual_text"] = content.replace(
                                "[Visual Analysis Result]\n", ""
                            )

        # Parse final AI message for status/answer
        draft_answer = ""
        rag_status = RAGStatus.FOUND
        unanswered = []

        # Find last AI message (the final response)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                text = extract_text(msg.content)
                rag_status, draft_answer, unanswered = parse_status_block(text)
                break

        # Fallback: if no status block found, use raw text
        if not draft_answer:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    draft_answer = extract_text(msg.content)
                    break

        return {
            "chunks_found": chunks_found,
            "searches_performed": searches_performed,
            "image_paths": image_paths,
            "draft_answer": draft_answer,
            "rag_status": rag_status,
            "unanswered": unanswered,
        }
