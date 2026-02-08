"""
Multi-Agent RAG Workflow with ReAct Agent and Thinking Levels

Architecture:
    User Query → Glossary Expansion → Regex Filter → RAG Agent (ReAct) → Verifier → Answer

Graph Structure:
    filter → rag_agent → verifier → format_final
           → direct_response (chitchat/out_of_scope)

    Revision: verifier (needs_revision) → rag_agent (max 1)
"""

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from config.settings import settings
from src.agent_tools import search_documents, set_global_retriever, visual_proof
from src.llm_factory import get_gemini_llm, get_llm
from src.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# --- Constants ---
THINKING_BUDGET = 8192  # ReAct agent: adaptive thinking (ceiling, not fixed)
THINKING_VERIFIER = 1024  # Verifier: moderate reasoning
MAX_REVISIONS = 1

GLOSSARY_PATH = Path(__file__).parent.parent / "config" / "term_glossary.yaml"

# Regex patterns for chitchat / out_of_scope detection (no LLM needed)
CHITCHAT_PATTERNS = re.compile(
    r"^\s*("
    r"привет\w*|здравствуй\w*|добрый\s+(день|вечер|утро)|"
    r"спасибо|благодар|пока\b|до свидания|"
    r"как дела|что (ты )?(умеешь|можешь|знаешь)|"
    r"кто ты|расскажи о себе|помоги\s*$|помощь\s*$|"
    r"hi\b|hello|thanks|bye"
    r")\s*[?!.]*\s*$",
    re.IGNORECASE,
)

OUT_OF_SCOPE_PATTERNS = re.compile(
    r"^\s*("
    r"какая погода|расскажи (анекдот|шутку|историю)|"
    r"(сколько|какой)\s+(стоит|цена)|"
    r"напиши (стих|код|программу|песню)|"
    r"переведи\s|"
    r"что такое (любовь|счастье|смысл жизни)"
    r")",
    re.IGNORECASE,
)


# --- Glossary ---
def _load_glossary(path: Path = GLOSSARY_PATH) -> dict[str, str]:
    """Load term glossary: {lowercase_short_term: official_term}."""
    if not path.exists():
        logger.warning("Term glossary not found at %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        terms = data.get("terms", {}) if data else {}
        return {key.lower(): val["official"] for key, val in terms.items() if "official" in val}
    except Exception as e:
        logger.error("Failed to load glossary: %s", e)
        return {}


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


def _expand_query(query: str, glossary: dict[str, str]) -> str:
    """Expand unofficial abbreviations in query using glossary.

    Appends glossary block for each match found.
    Handles Russian declensions via stem-based matching.
    Example: "программы А" → "программы А\\n\\n[Глоссарий: программа а → ...]"
    """
    if not glossary:
        return query
    expansions = []
    for short_term, official in glossary.items():
        pattern = _make_term_pattern(short_term)
        if pattern.search(query):
            expansions.append(f"{short_term} → {official}")
    if expansions:
        return query + "\n\n[Глоссарий: " + "; ".join(expansions) + "]"
    return query


def _classify_query(query: str) -> str:
    """Classify query as chitchat, out_of_scope, or rag using regex patterns.

    Returns: 'chitchat', 'out_of_scope', or 'rag'
    """
    # Strip glossary block for classification
    clean_query = re.sub(r"\n\n\[Глоссарий:.*\]$", "", query, flags=re.DOTALL).strip()
    if CHITCHAT_PATTERNS.match(clean_query):
        return "chitchat"
    if OUT_OF_SCOPE_PATTERNS.match(clean_query):
        return "out_of_scope"
    return "rag"


# --- Enums ---
class RouteType(str, Enum):
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    RAG = "rag"


class RAGStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    PARTIAL = "PARTIAL"


class VerifyStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"


# --- State ---
class ChunkInfo(TypedDict):
    """Structured chunk information."""

    content: str
    source: str
    page_no: Optional[int]
    bbox: Optional[List[float]]
    visual_text: Optional[str]


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


# --- Parsers ---
def _parse_json_from_response(raw: str) -> dict:
    """Extract JSON from LLM response with multiple fallback strategies."""
    # 1. Try markdown code blocks
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if code_match:
        raw_json = code_match.group(1)
    else:
        # 2. Fallback: find first {...}
        brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        raw_json = brace_match.group(1) if brace_match else "{}"

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return {}


def _parse_search_results(search_output: str) -> List[ChunkInfo]:
    """Parse search_documents output into structured ChunkInfo list."""
    chunks = []

    result_pattern = re.compile(
        r"\[Result \d+\] File: ([^\|]+)\| Page: ([^\|]+)\| BBox: ([^\n]+)\n"
        r"(?:Extended Context:\n)?(.*?)(?:\(IDs: [^\)]+\))?(?=\[Result|\Z)",
        re.DOTALL,
    )

    for match in result_pattern.finditer(search_output):
        source = match.group(1).strip()
        page_str = match.group(2).strip()
        bbox_str = match.group(3).strip()
        content = match.group(4).strip() if match.group(4) else ""

        try:
            page_no = int(page_str) if page_str != "N/A" else None
        except ValueError:
            page_no = None

        bbox = None
        if bbox_str and bbox_str not in ("N/A", "None"):
            try:
                bbox = json.loads(bbox_str.replace("'", '"'))
            except json.JSONDecodeError:
                pass

        chunks.append(
            ChunkInfo(
                content=content,
                source=source,
                page_no=page_no,
                bbox=bbox,
                visual_text=None,
            )
        )

    # Fallback: if no structured results, treat whole output as single chunk
    if (
        not chunks
        and search_output
        and "No relevant documents found" not in search_output
    ):
        chunks.append(
            ChunkInfo(
                content=search_output,
                source="unknown",
                page_no=None,
                bbox=None,
                visual_text=None,
            )
        )

    return chunks


def _parse_status_block(text: str) -> tuple[RAGStatus, str, list[str]]:
    """Parse ===STATUS===, ===ANSWER===, and ===UNANSWERED=== blocks from agent output.

    Returns:
        (rag_status, answer_text, unanswered_list)
    """
    # Parse status
    status = RAGStatus.FOUND  # default
    status_match = re.search(r"===STATUS===\s*\n\s*(\w+)", text)
    if status_match:
        raw_status = status_match.group(1).strip().upper()
        if raw_status in (s.value for s in RAGStatus):
            status = RAGStatus(raw_status)

    # Parse answer
    answer = text  # fallback: whole text
    answer_match = re.search(r"===ANSWER===\s*\n(.*?)(?:===UNANSWERED===|\Z)", text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Parse unanswered
    unanswered = []
    unanswered_match = re.search(r"===UNANSWERED===\s*\n(.*?)$", text, re.DOTALL)
    if unanswered_match:
        for line in unanswered_match.group(1).strip().splitlines():
            line = line.strip().lstrip("- ")
            if line:
                unanswered.append(line)

    return status, answer, unanswered


class MultiAgentRAGWorkflow:
    """
    RAG workflow with a single ReAct agent and thinking levels.

    The agent autonomously decides when to search, decompose,
    and use visual_proof based on the merged prompt.

    Graph: filter → rag_agent → verifier → format_final
                  → direct_response (chitchat/out_of_scope)

    Supports:
    - Gemini: Flash with adaptive thinking budget (up to 8192)
    - OpenAI: Uses configured model for all steps (no thinking budgets)
    """

    def __init__(self, retriever: BaseRetriever, llm_provider: str = "gemini"):
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()
        set_global_retriever(retriever)

        # LLMs based on provider
        if self.llm_provider == "gemini":
            self.rag_llm = get_gemini_llm(
                settings.GEMINI_FAST_MODEL,
                thinking_budget=THINKING_BUDGET,
            )
            self.verifier_llm = get_gemini_llm(
                settings.GEMINI_FAST_MODEL,
                thinking_budget=THINKING_VERIFIER,
                response_mime_type="application/json",
            )
        else:  # openai fallback
            openai_llm = get_llm()
            self.rag_llm = openai_llm
            self.verifier_llm = openai_llm

        self.prompt_manager = PromptManager()
        self.glossary = _load_glossary()
        self.compiled_workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(RAGState)

        graph.add_node("filter", self._filter_node)
        graph.add_node("direct_response", self._direct_response_node)
        graph.add_node("rag_agent", self._rag_agent_node)
        graph.add_node("verifier", self._verifier_node)
        graph.add_node("format_final", self._format_final_node)

        graph.set_entry_point("filter")

        graph.add_conditional_edges(
            "filter",
            self._route_after_filter,
            {
                "direct_response": "direct_response",
                "rag_agent": "rag_agent",
            },
        )
        graph.add_edge("direct_response", END)

        graph.add_edge("rag_agent", "verifier")

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

    def invoke(self, query: str) -> Dict:
        """
        Process a user query through the RAG workflow.

        Args:
            query: User's question

        Returns:
            Dict with route_type, final_answer, and other state fields
        """
        # Expand query with glossary before entering the graph
        expanded_query = _expand_query(query, self.glossary)
        if expanded_query != query:
            logger.info("Query expanded: %r → %r", query, expanded_query)

        initial_state: RAGState = {
            "query": expanded_query,
            "route_type": RouteType.RAG,
            "direct_response": None,
            "searches_performed": [],
            "chunks_found": [],
            "image_paths": [],
            "draft_answer": "",
            "unanswered": [],
            "rag_status": RAGStatus.FOUND,
            "verify_status": VerifyStatus.APPROVED,
            "verify_issues": [],
            "revision_count": 0,
            "final_answer": "",
        }

        final_state = self.compiled_workflow.invoke(
            initial_state, {"recursion_limit": 25}
        )

        return final_state

    # --- Node implementations ---

    def _filter_node(self, state: RAGState) -> dict:
        """Regex-based filter: classify as chitchat/out_of_scope/rag."""
        classification = _classify_query(state["query"])
        return {"route_type": classification}

    def _direct_response_node(self, state: RAGState) -> dict:
        """Return direct response for chitchat/out_of_scope."""
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

        tools = [search_documents, visual_proof]

        agent = create_react_agent(
            model=self.rag_llm,
            tools=tools,
            prompt=system_prompt,
        )
        result = agent.invoke({"messages": [("user", user_msg)]})
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
        parsed = _parse_json_from_response(str(response.content).strip())

        return {
            "verify_status": parsed.get("status", "approved"),
            "verify_issues": parsed.get("issues", []),
            "revision_count": state.get("revision_count", 0) + 1,
        }

    def _format_final_node(self, state: RAGState) -> dict:
        """Format the final answer."""
        answer = state.get("draft_answer", "")

        if state.get("verify_status") == VerifyStatus.NEEDS_REVISION:
            answer += "\n\n⚠️ Ответ предоставлен с оговорками, рекомендуется дополнительная проверка."

        return {"final_answer": answer}

    # --- Routing functions ---

    def _route_after_filter(self, state: RAGState) -> str:
        route_type = state.get("route_type", "")
        if route_type in (RouteType.CHITCHAT, RouteType.OUT_OF_SCOPE, "chitchat", "out_of_scope"):
            return "direct_response"
        return "rag_agent"

    def _route_after_verify(self, state: RAGState) -> str:
        if state.get("verify_status") in (VerifyStatus.APPROVED, "approved"):
            return "format_final"
        if state.get("revision_count", 0) > MAX_REVISIONS:
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
                        searches_performed.append({"query": query_arg, "results_count": 0})

            # Extract chunks from search results
            if isinstance(msg, ToolMessage):
                if msg.name == "search_documents":
                    parsed_chunks = _parse_search_results(str(msg.content))
                    # Update results_count for the last search
                    if searches_performed:
                        searches_performed[-1]["results_count"] = len(parsed_chunks)
                    chunks_found.extend(parsed_chunks)

                elif msg.name == "visual_proof":
                    content = str(msg.content)
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
                text = str(msg.content)
                rag_status, draft_answer, unanswered = _parse_status_block(text)
                break

        # Fallback: if no status block found, use raw text
        if not draft_answer:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    draft_answer = str(msg.content)
                    break

        return {
            "chunks_found": chunks_found,
            "searches_performed": searches_performed,
            "image_paths": image_paths,
            "draft_answer": draft_answer,
            "rag_status": rag_status,
            "unanswered": unanswered,
        }
