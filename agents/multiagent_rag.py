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
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_stream_writer # Import get_stream_writer

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
    detect_incomplete_chunk, # Import detect_incomplete_chunk
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

# --- Helper for visual proof processing ---
def _process_visual_proof(chunks: list[dict], visual_proof_tool, writer) -> list[dict]:
    """
    Обрабатывает visual proof для списка чанков.
    Отправляет статусы через writer.
    """
    processed_chunks = []
    for i, chunk in enumerate(chunks[:settings.MAX_VISUAL_PROOFS]):  # Process top N chunks
        # Check for metadata first, then use heuristic
        needs_analyze = (
            chunk.get("metadata", {}).get("needs_visual_analyze")
            or detect_incomplete_chunk(chunk["content"])
        )
        
        # Default values
        chunk["visual_proof_mode"] = "none"
        chunk["image_path"] = None

        # Ensure chunk has necessary fields for visual proof tool
        file_name = chunk.get("source")
        page_no = chunk.get("page_no")
        bbox = chunk.get("bbox")

        if not all([file_name, page_no, bbox]):
            # Cannot perform visual proof without complete metadata
            writer({"status": f"⚠️ Фрагмент {i+1}: неполные метаданные для визуальной проверки"})
            processed_chunks.append(chunk)
            continue

        if needs_analyze:
            # Determine the reason for analysis (can be expanded)
            reason_labels = {
                "table_fragment": "содержит таблицу",
                "incomplete_sentence": "обрезан",
                "list_not_from_start": "список не с начала",
                "continuation_marker": "маркер продолжения",
                "header_only": "только заголовок",
            }
            # For now, just indicate it's incomplete if heuristic detected it
            label = "неполный" if detect_incomplete_chunk(chunk["content"]) else "анализ"
            if chunk.get("metadata", {}).get("reasons"):
                label = reason_labels.get(chunk["metadata"]["reasons"][0], label)

            writer({"status": f"🖼️ Фрагмент {i+1} {label} — анализирую изображение..."})
            vp_result = visual_proof_tool.invoke({
                "file_name": file_name,
                "page_no": page_no,
                "bbox": bbox,
                "mode": "analyze"
            })
            extracted_text = extract_text(vp_result).replace("[Visual Analysis Result]\n", "").strip()
            chunk["content"] = extracted_text # Update chunk content with VLM result
            chunk["visual_text"] = extracted_text # Store VLM result separately as well
            chunk["visual_proof_mode"] = "analyze"
            chunk["image_path"] = f"Visual analysis for {file_name} page {page_no}" # Placeholder for image path for analyze mode
            writer({"status": "🖼️ Текст извлечён из изображения"})
        else:
            writer({"status": f"✅ Фрагмент {i+1} полный"})
            vp_result = visual_proof_tool.invoke({
                "file_name": file_name,
                "page_no": page_no,
                "bbox": bbox,
                "mode": "show"
            })
            chunk["visual_proof_mode"] = "show"
            chunk["image_path"] = extract_text(vp_result) # Actual image path
        
        processed_chunks.append(chunk)
    
    return processed_chunks

# --- State ---
class RAGState(TypedDict):
    """State for the Multi-Agent RAG workflow."""

    query: str
    route_type: RouteType
    direct_response: Optional[str]
    searches_performed: list[dict]  # [{query, results_count}]
    chunks_found: list[ChunkInfo]
    image_paths: list[str]
    visual_proof_mode: Optional[str] # New field to track visual proof mode
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
            self.rag_complex_llm = get_gemini_llm( # New LLM for complex RAG
                settings.GEMINI_FAST_MODEL,
                thinking_budget=settings.THINKING_BUDGET,
                temperature=0.2, # Lower temperature for more focused responses
            )
        else:  # openai fallback
            openai_llm = get_llm()
            self.rag_llm = openai_llm
            self.verifier_llm = openai_llm
            self.rag_complex_llm = openai_llm # New LLM for complex RAG


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
        graph.add_node("rag_simple", self._rag_simple_node) # New node
        graph.add_node("escalation", self._escalation_node) # New node
        graph.add_node("rag_complex", self._rag_complex_node) # New node
        graph.add_node("verifier", self._verifier_node)
        graph.add_node("format_final", self._format_final_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            self._route_after_router, # This will be modified
            {
                "direct_response": "direct_response",
                "rag_simple": "rag_simple", # Route to new simple RAG
                "rag_complex": "rag_complex", # Route to new complex RAG
            },
        )
        graph.add_edge("direct_response", END)

        graph.add_conditional_edges( # New conditional edges for rag_simple
            "rag_simple",
            self._route_after_rag_simple,
            {
                "escalation": "escalation",
                "verifier": "verifier",
                "format_final": "format_final",
            },
        )
        graph.add_edge("escalation", "rag_complex") # New edge
        graph.add_edge("rag_complex", "verifier") # New edge

        graph.add_conditional_edges(
            "verifier",
            self._route_after_verify,
            {
                "format_final": "format_final",
                "rag_simple": "rag_simple", # Route back to simple for revisions
                "rag_complex": "rag_complex", # Route back to complex for revisions
            },
        )
        graph.add_edge("format_final", END)

        return graph.compile()

    def stream_events(self, query: str):
        """
        Process a user query and yield status updates + final answer.

        Yields dicts with "type" key:
          - {"type": "status", "text": "..."} — progress updates
          - {"type": "final", "answer": "...", "chunks_found": [...], "image_paths": [...]}
        """
        # 0. Check Semantic Cache
        if self.cache:
            cached_answer = self.cache.get(query)
            if cached_answer:
                yield {"type": "status", "text": "🔎 Ответ найден в кеше!"}
                yield {
                    "type": "final",
                    "answer": cached_answer,
                    "chunks_found": [],
                    "image_paths": [],
                }
                return

        expanded_query = _expand_query(query)

        initial_state: RAGState = {
            "query": expanded_query,
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
            "is_routed": False,
        }

        self.tool_ctx.search_call_count = 0
        self.tool_ctx.visual_proof_call_count = 0

        final_answer = ""

        # stream_mode="custom" — single mode, no tuple wrapping
        for event in self.compiled_workflow.stream(
            initial_state, {"recursion_limit": 50}, stream_mode="custom"
        ):
            if not isinstance(event, dict):
                continue

            if event.get("type") == "final":
                final_answer = event.get("answer", "")
                yield event
            elif "status" in event and "type" not in event:
                yield {"type": "status", "text": event["status"]}
            else:
                yield event

        # Update cache (skip answers with verification warnings)
        if self.cache and final_answer:
            if "⚠️ Ответ предоставлен с оговорками" not in final_answer:
                self.cache.add(query, final_answer)

    def _get_tool_by_name(self, tool_name: str):
        for tool_obj in self.tools:
            if tool_obj.name == tool_name:
                return tool_obj
        return None

    def _search_documents_tool_wrapper(self, query: str) -> list[ChunkInfo]:
        search_tool = self._get_tool_by_name("search_documents")
        if not search_tool:
            logger.error("search_documents tool not found.")
            return []
        
        try:
            result_str = search_tool.invoke({"query": query})
            chunks = parse_search_results(result_str)
            return chunks
        except Exception as e:
            logger.error(f"Error invoking search_documents tool: {e}")
            return []

    def _build_search_query(self, state: RAGState) -> str:
        # For simple RAG, just use the main query
        return state["query"]

    def _rephrase_query(self, state: RAGState, search_results: list[ChunkInfo]) -> str:
        # Use LLM to rephrase query if initial search failed or was not ideal
        prompt = self.prompt_manager.render(
            "applicability_retriever",
            question=state["query"],
            context=search_results # Provide context of initial failed search
        )
        response = self.rag_llm.invoke([HumanMessage(content=prompt)])
        return extract_text(response.content).strip()

    def _build_rag_simple_context(self, state: RAGState, chunks: list[ChunkInfo]) -> str:
        # Build prompt for rag_simple LLM
        prompt = self.prompt_manager.render(
            "rag_simple",
            query=state["query"],
            context=chunks
        )
        return prompt
    
    def _decompose_query(self, query: str, llm) -> list[str]:
        # Placeholder for query decomposition
        prompt = self.prompt_manager.render(
            "applicability_retriever",
            question=query
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        # Expecting a list of subquestions, potentially JSON parsed
        # For now, a very simple decomposition based on sentences or just returning the query
        text_content = extract_text(response.content)
        # Attempt to parse as JSON list, otherwise split by sentence.
        try:
            parsed = json.loads(text_content)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback to splitting by sentence or just returning the query if no clear decomposition
        sentences = re.split(r'(?<=[.!?])\s+', text_content)
        return [s.strip() for s in sentences if s.strip()] or [query]
            
    def _rephrase_subquestion(self, subquestion: str, all_chunks: list[ChunkInfo]) -> str:
        # Placeholder for rephrasing subquestions, similar to _rephrase_query but for subquestions
        prompt = self.prompt_manager.render(
            "applicability_retriever",
            question=subquestion,
            context=all_chunks
        )
        response = self.rag_complex_llm.invoke([HumanMessage(content=prompt)])
        return extract_text(response.content).strip()

    def _filter_chunks_for_context(self, chunks: list[ChunkInfo]) -> list[ChunkInfo]:
        # Placeholder for filtering chunks, can use another LLM call or rule-based
        # For now, a simple filter to remove empty chunks or very short ones
        return [c for c in chunks if c.get("content") and len(c["content"]) > settings.MIN_CHUNK_LENGTH_FOR_FILTERING]

    def _build_rag_complex_context(self, state: RAGState, chunks: list[ChunkInfo], subquestions: list[str]) -> str:
        # Build prompt for rag_complex LLM
        prompt = self.prompt_manager.render(
            "rag_complex",
            query=state["query"],
            subquestions=subquestions,
            context=chunks,
            searches_performed=state.get("searches_performed", []),
        )
        return prompt

    @staticmethod
    def _max_similarity(results: list[ChunkInfo]) -> float:
        """Return the maximum similarity score across all results."""
        if not results:
            return 0.0
        return max(r.get("similarity", 0.0) or 0.0 for r in results)

    # --- Node implementations ---

    def _router_node(self, state: RAGState) -> dict:
        """LLM-based router: classify query."""
        # If already routed (via speculative execution), skip re-running
        if state.get("is_routed"):
            return {}

        writer = get_stream_writer()
        writer({"status": "🔍 Классифицирую запрос..."})

        result = self.router_agent.route(state["query"])

        updates = {"route_type": result["type"]}
        if result["response"]:
            updates["direct_response"] = result["response"]
        
        route_labels = {
            RouteType.RAG_SIMPLE: "📂 Простой вопрос — ищу в документах",
            RouteType.RAG_COMPLEX: "📂 Составной вопрос — провожу развёрнутый анализ",
            RouteType.CHITCHAT: f"💬 {result['type']}",
            RouteType.OUT_OF_SCOPE: f"💬 {result['type']}",
        }
        writer({"status": route_labels.get(result['type'], "🤔 Думаю...")}) # Default status if route type is unknown
        
        return updates

    def _direct_response_node(self, state: RAGState) -> dict:
        """Return direct response for chitchat/out_of_scope."""
        writer = get_stream_writer()

        if state.get("direct_response"):
            response = state["direct_response"]
        elif state.get("route_type") in (RouteType.CHITCHAT, "chitchat"):
            response = "Здравствуйте! Я готов помочь с вопросами по охране труда и промышленной безопасности."
        else:
            response = (
                "Этот вопрос не относится к охране труда и промышленной безопасности."
            )

        writer({
            "type": "final",
            "answer": response,
            "chunks_found": [],
            "image_paths": [],
        })
        return {"final_answer": response}

    def _rag_simple_node(self, state: RAGState) -> dict:
        writer = get_stream_writer()
        
        # --- Preprocessing ---
        query = self._build_search_query(state)
        
        # --- Search 1 ---
        writer({"status": f"🔎 Поиск: \"{query}\""})
        results = self._search_documents_tool_wrapper(query)
        
        searches = [{"query": query, "results_count": len(results)}]
        
        visual_proof_tool = self._get_tool_by_name("visual_proof")
        if not visual_proof_tool:
            logger.error("visual_proof tool not found.")
            return {"chunks_found": [], "searches_performed": searches, "rag_status": RAGStatus.NOT_FOUND,}

        if results and self._max_similarity(results) >= settings.SIMILARITY_THRESHOLD_ACCEPTANCE:
            best_score = self._max_similarity(results)
            top = results[0]
            writer({
                "status": f"📄 Найдено {len(results)} фрагментов "
                          f"(лучший: {top.get('source', 'unknown')}, "
                          f"score: {best_score:.2f})"
            })
        else:
            # --- Search 2 (rephrasing) ---
            query2 = self._rephrase_query(state, results)
            writer({"status": f"🔎 Уточняю: \"{query2}\""})
            results2 = self._search_documents_tool_wrapper(query2)
            searches.append({"query": query2, "results_count": len(results2)})

            if results2 and self._max_similarity(results2) >= settings.SIMILARITY_THRESHOLD_ACCEPTANCE:
                results = results2
                writer({"status": f"📄 Найдено: {results2[0].get('source', 'unknown')}"})
            else:
                writer({"status": "⚠️ Релевантных фрагментов не найдено"})
                return {
                    "chunks_found": [],
                    "searches_performed": searches,
                    "rag_status": RAGStatus.NOT_FOUND,
                }
        
        # --- Visual proof ---
        chunks = _process_visual_proof(results, visual_proof_tool, writer)
        
        # --- Generate answer ---
        writer({"status": "📝 Формирую ответ..."})
        draft = self.rag_llm.invoke(
            [HumanMessage(content=self._build_rag_simple_context(state, chunks))]
        )
        
        return {
            "chunks_found": chunks,
            "searches_performed": searches,
            "draft_answer": extract_text(draft.content),
            "rag_status": RAGStatus.FOUND,
        }

    def _escalation_node(self, state: RAGState) -> dict:
        """Placeholder for Escalation node."""
        writer = get_stream_writer()
        writer({"status": "🔄 Простой поиск не дал результатов. Провожу углублённый анализ документов..."})
        return {"escalated_from_simple": True}

    def _rag_complex_node(self, state: RAGState) -> dict:
        writer = get_stream_writer()
        
        # --- Decomposition ---
        writer({"status": "🧩 Разбиваю вопрос на подвопросы..."})
        subquestions = self._decompose_query(state["query"], self.rag_complex_llm)
        writer({"status": f"🧩 Выделено {len(subquestions)} подвопросов"})
        
        # --- Context for escalation ---
        prev_searches = state.get("searches_performed", [])
        prev_queries = {s["query"] for s in prev_searches}
        
        # --- Search for each subquestion ---
        all_chunks = []
        all_searches = list(prev_searches)
        total = len(subquestions)
        
        visual_proof_tool = self._get_tool_by_name("visual_proof")
        if not visual_proof_tool:
            logger.error("visual_proof tool not found.")
            return {"chunks_found": [], "searches_performed": all_searches, "rag_status": RAGStatus.NOT_FOUND,}

        for i, sq in enumerate(subquestions, 1):
            # Skip already performed queries (from escalation)
            if sq in prev_queries:
                writer({"status": f"⏭️ [{i}/{total}] Уже искали: \"{sq}\""})
                continue
            
            # Iteration 1: exact terms
            writer({"status": f"🔎 [{i}/{total}] Ищу: \"{sq}\""})
            results = self._search_documents_tool_wrapper(sq)
            all_searches.append({"query": sq, "results_count": len(results)})
            
            if results and self._max_similarity(results) >= settings.SIMILARITY_THRESHOLD_ACCEPTANCE:
                all_chunks.extend(results[:3]) # Take top 3 relevant chunks
                writer({"status": f"📄 [{i}/{total}] Найдено: {results[0].get('source', 'unknown')}"})
            else:
                # Iteration 2: synonyms + metadata
                sq2 = self._rephrase_subquestion(sq, all_chunks) # Rephrase based on current context
                writer({"status": f"⚠️ [{i}/{total}] Не найдено, пробую: \"{sq2}\""})
                results2 = self._search_documents_tool_wrapper(sq2)
                all_searches.append({"query": sq2, "results_count": len(results2)})

                if results2 and self._max_similarity(results2) >= settings.SIMILARITY_THRESHOLD_ACCEPTANCE:
                    all_chunks.extend(results2[:3])
                    writer({"status": f"📄 [{i}/{total}] Найдено: {results2[0].get('source', 'unknown')}"})
                else:
                    writer({"status": f"❌ [{i}/{total}] Данные не обнаружены"})
        
        # --- Visual proof ---
        if all_chunks:
            chunks = _process_visual_proof(all_chunks, visual_proof_tool, writer)
        else:
            chunks = []
        
        # --- Filtering ---
        if chunks:
            writer({"status": "🔬 Фильтрую нерелевантное..."})
            chunks = self._filter_chunks_for_context(chunks)
        
        # --- Generate answer ---
        writer({"status": "📝 Собираю ответ из найденных фрагментов..."})
        draft = self.rag_complex_llm.invoke(
            [HumanMessage(content=self._build_rag_complex_context(state, chunks, subquestions))]
        )
        
        return {
            "chunks_found": chunks,
            "searches_performed": all_searches,
            "subquestions": subquestions,
            "draft_answer": extract_text(draft.content),
            "rag_status": RAGStatus.FOUND if chunks else RAGStatus.NOT_FOUND,
        }

    def _verifier_node(self, state: RAGState) -> dict:
        """Verify draft answer against found chunks."""
        writer = get_stream_writer()
        writer({"status": "✅ Проверяю точность ответа..."})

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
            issues_count = len(parsed.get("issues", []))
            writer({"status": f"🔄 Найдено {issues_count} замечаний, корректирую..."})
        else:
            writer({"status": "✅ Ответ проверен"})

        return {
            "verify_status": status,
            "verify_issues": parsed.get("issues", []),
            "revision_count": revision_count,
        }

    def _format_final_node(self, state: RAGState) -> dict:
        """Format the final answer."""
        writer = get_stream_writer()
        answer = state.get("draft_answer", "")

        # Strip <thinking> blocks from LLM output
        answer = re.sub(r"<thinking>.*?</thinking>\s*", "", answer, flags=re.DOTALL)

        # Strip leaked status markers from Gemini responses
        status_idx = answer.find("===STATUS===")
        if status_idx != -1:
            answer = answer[:status_idx].rstrip()

        if state.get("verify_status") == VerifyStatus.NEEDS_REVISION:
            answer += "\n\n⚠️ Ответ предоставлен с оговорками, рекомендуется дополнительная проверка."

        writer({
            "type": "final",
            "answer": answer,
            "chunks_found": state.get("chunks_found", []),
            "image_paths": state.get("image_paths", []),
        })
        return {"final_answer": answer}

    # --- Routing functions ---

    def _route_after_router(self, state: RAGState) -> str:
        route_type = state.get("route_type", "")
        if route_type in (
            RouteType.CHITCHAT,
            RouteType.OUT_OF_SCOPE,
        ):
            return "direct_response"
        elif route_type == RouteType.RAG_SIMPLE:
            return "rag_simple"
        return "rag_complex" # Default to complex if not simple or direct

    def _route_after_rag_simple(self, state: RAGState) -> str:
        """Decide whether to escalate, verify or trust the answer after simple RAG."""
        chunks = state.get("chunks_found", [])

        if state.get("rag_status") == RAGStatus.NOT_FOUND or not chunks:
            return "escalation"

        # Check visual proof
        if any(c.get("visual_text") for c in chunks):
            return "verifier"

        # Check source consistency (more than one source usually means need for complex verification)
        sources = {c.get("source") for c in chunks if c.get("source")}
        if len(sources) > 1:
            return "verifier"

        # Check similarity score (high confidence allows skipping verification)
        top_chunk = chunks[0]
        score = top_chunk.get("similarity", 0.0)
        if score > settings.SIMILARITY_THRESHOLD_FOR_VERIFIER_SKIP:
            logger.info(f"Skipping verification! High confidence: {score}")
            return "format_final"
        
        return "verifier"

    def _route_after_verify(self, state: RAGState) -> str:
        if state.get("verify_status") in (VerifyStatus.APPROVED, "approved"):
            return "format_final"
        if state.get("revision_count", 0) > settings.MAX_REVISIONS:
            return "format_final"  # max revisions reached
        
        # Route back to the appropriate RAG node for revision
        # This assumes the revision logic should go to the 'complex' if it was already there,
        # or simple if it was originally simple. This might need more sophisticated state tracking.
        if state.get("escalated_from_simple"): # Assuming a state variable to track this
            return "rag_complex"
        return "rag_simple"

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
