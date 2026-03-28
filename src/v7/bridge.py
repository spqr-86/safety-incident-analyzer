"""Bridge: adapt existing ChromaDB vector store for v7 pipeline.

Responsibilities:
1. Wrap ChromaDB's similarity_search_with_score -> v7 dict format
2. Build BM25 corpus from ChromaDB docs
3. Inject search functions into rag_simple / rag_complex nodes
4. Inject FlashRank reranker into rag_complex
5. Inject LLM-backed verify, rewrite, and generate functions
"""

from __future__ import annotations

import logging
from typing import Callable, List

from langchain_core.messages import HumanMessage

from src.llm_factory import get_gemini_llm
from src.parsers import extract_text, parse_json_from_response
from src.v7.nlp_core import init_bm25_index
from src.v7.nodes import generate_answer as generate_answer_mod
from src.v7.nodes import llm_verifier as llm_verifier_mod
from src.v7.nodes import rag_complex as rag_complex_mod
from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes import rewriter as rewriter_mod
from src.v7.nodes.llm_verifier import VERIFIER_SYSTEM_PROMPT
from src.v7.nodes.utils import extract_doc_identifiers
from src.v7.state_types import VerificationResult

logger = logging.getLogger(__name__)


def make_rerank_fn(
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    cache_dir: str = ".flashrank_cache",
) -> Callable[[str, List[dict], int], List[dict]]:
    """Create a FlashRank reranker function for v7 rag_complex.

    Signature: fn(query, passages, top_k) -> passages (reranked, top_k items).
    Passages must have 'text' key; score is updated with FlashRank score.
    """
    from flashrank import Ranker, RerankRequest

    ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def _rerank(query: str, passages: List[dict], top_k: int) -> List[dict]:
        if not passages:
            return passages
        rerank_request = RerankRequest(
            query=query,
            passages=[
                {"id": i, "text": p.get("text", "")} for i, p in enumerate(passages)
            ],
        )
        results = ranker.rerank(rerank_request)
        # results: list of dicts with 'id', 'score', 'text'
        reranked = []
        for r in results[:top_k]:
            orig = passages[r["id"]]
            reranked.append({**orig, "score": round(float(r["score"]), 4)})
        return reranked

    return _rerank


def make_vector_search_fn(vector_store) -> Callable[..., List[dict]]:
    """Create a v7-compatible vector search function from ChromaDB store.

    v7 interface: fn(query, filters=None, top_k=12, **kwargs) -> list[dict]
    Each dict has: text, metadata, score.
    """

    def _search(
        query: str,
        filters: dict | None = None,
        top_k: int = 12,
        **kwargs,
    ) -> List[dict]:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, distance in docs_and_scores:
            # ChromaDB returns L2 distance (0..inf). Convert to similarity (0..1).
            similarity = 1.0 / (1.0 + distance)
            results.append(
                {
                    "text": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": round(similarity, 4),
                }
            )
        return results

    return _search


def make_section_fetch_fn(
    vector_store,
    max_section_chunks: int = 30,
) -> Callable[[List[dict]], List[dict]]:
    """Create a section-aware expander from ChromaDB store.

    Takes the top anchor passage, extracts parent_section + source from its metadata,
    and fetches all chunks from that section (up to max_section_chunks).
    Returns passages not already in the input list.
    """

    def _fetch_section(passages: List[dict]) -> List[dict]:
        if not passages:
            return []
        anchor_meta = passages[0].get("metadata", {})
        section = anchor_meta.get("parent_section", "")
        source = anchor_meta.get("source", "")
        if not section or not source:
            return []
        try:
            col = vector_store._collection
            results = col.get(
                where={
                    "$and": [
                        {"parent_section": {"$eq": section}},
                        {"source": {"$eq": source}},
                    ]
                },
                include=["documents", "metadatas"],
                limit=max_section_chunks,
            )
            extra = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                extra.append(
                    {
                        "text": doc,
                        "metadata": dict(meta),
                        "score": 0.0,  # no vector score for fetched chunks
                    }
                )
            return extra
        except Exception as exc:
            logger.warning("section_fetch failed: %s", exc)
            return []

    return _fetch_section


def make_verify_fn(llm) -> Callable[..., VerificationResult]:
    """Create a v7-compatible verify function backed by Gemini LLM."""

    def _verify(
        original_query: str, active_query: str, passages: List[dict]
    ) -> VerificationResult:
        passages_text = "\n\n".join(
            f"[{i + 1}] (score={p.get('score', 'N/A')}): {p.get('text', '')}"
            for i, p in enumerate(passages)
        )
        prompt = (
            f"{VERIFIER_SYSTEM_PROMPT}\n\n"
            f"Запрос пользователя: {original_query}\n"
            f"Активный запрос: {active_query}\n\n"
            f"Найденные passages ({len(passages)}):\n{passages_text}"
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_text = extract_text(response.content)
            data = parse_json_from_response(raw_text)
            if not data or "verdict" not in data:
                raise ValueError("No verdict in LLM response")
            return VerificationResult(
                verdict=data["verdict"],
                reason=data.get("reason", ""),
                rewrite_hint=data.get("rewrite_hint", ""),
                missing_aspects=data.get("missing_aspects", []),
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception as exc:
            logger.warning("LLM verify failed: %s", exc)
            return VerificationResult(
                verdict="escalate",
                reason=f"LLM parse error: {type(exc).__name__}",
                missing_aspects=[],
                confidence=0.0,
            )

    return _verify


def make_rewrite_fn(llm) -> Callable[..., str]:
    """Create a v7-compatible rewrite function backed by Gemini LLM."""

    def _rewrite(
        original_query: str,
        active_query: str,
        rewrite_hint: str,
        missing_aspects: List[str],
    ) -> str:
        protected_ids = extract_doc_identifiers(original_query)
        aspects_str = ", ".join(missing_aspects) if missing_aspects else ""
        prompt = (
            "Переформулируй поисковый запрос для поиска в базе нормативных документов.\n"
            "Сохрани ВСЕ номера документов (ГОСТ, СП, СНиП и т.д.) из исходного запроса.\n\n"
            f"Исходный запрос: {original_query}\n"
            f"Текущий запрос: {active_query}\n"
            f"Подсказка: {rewrite_hint}\n"
            f"Недостающие аспекты: {aspects_str}\n\n"
            "Верни ТОЛЬКО переформулированный запрос, без пояснений."
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            rewritten = extract_text(response.content).strip()
            if not rewritten:
                raise ValueError("Empty rewrite response")
            # Protect doc identifiers
            for doc_id in protected_ids:
                if doc_id not in rewritten:
                    rewritten = f"{rewritten} [{doc_id}]"
            return rewritten
        except Exception as exc:
            logger.warning("LLM rewrite failed: %s, falling back to stub", exc)
            # Fallback to stub logic
            rewritten = (
                f"{original_query} ({aspects_str})" if aspects_str else original_query
            )
            for doc_id in protected_ids:
                if doc_id not in rewritten:
                    rewritten = f"{rewritten} [{doc_id}]"
            return rewritten

    return _rewrite


_GENERATE_SYSTEM_PROMPT = (
    "Ты — эксперт по нормативным документам в области охраны труда и промышленной "
    "безопасности. На основе предоставленных фрагментов нормативных документов дай "
    "точный, структурированный ответ на вопрос пользователя. "
    "Ссылайся на конкретные документы и пункты. "
    "Отвечай только на основе предоставленных фрагментов — не придумывай информацию."
)


def make_generate_fn(llm) -> Callable[[str, str, List[dict]], str]:
    """Create an LLM-backed answer generation function for v7 generate_answer node.

    Signature: fn(query, active_query, passages) -> answer_text.
    Falls back to concatenating passage texts on LLM error.
    """

    def _generate(query: str, active_query: str, passages: List[dict]) -> str:
        if not passages:
            return ""
        passages_text = "\n\n".join(
            f"[{i + 1}] {p.get('text', '')}" for i, p in enumerate(passages[:15])
        )
        prompt = (
            f"{_GENERATE_SYSTEM_PROMPT}\n\n"
            f"Вопрос: {query}\n\n"
            f"Найденные фрагменты ({len(passages[:15])}):\n{passages_text}\n\n"
            "Ответ:"
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = extract_text(response.content).strip()
            if not answer:
                raise ValueError("Empty generation response")
            return answer
        except Exception as exc:
            logger.warning("LLM generate failed: %s, falling back to stub", exc)
            return "\n\n".join(p.get("text", "") for p in passages[:10])

    return _generate


def init_v7_from_chroma(vector_store, llm_provider: str | None = "gemini") -> None:
    """Initialize v7 pipeline from existing ChromaDB vector store.

    1. Creates vector search wrapper
    2. Injects it into rag_simple and rag_complex nodes
    3. Builds BM25 index from full corpus
    4. Injects FlashRank reranker into rag_complex
    5. Injects LLM-backed verify, rewrite, and generate functions (if provider available)
    """
    from config.settings import settings

    search_fn = make_vector_search_fn(vector_store)
    rag_simple_mod.set_vector_search(search_fn)
    rag_complex_mod.set_vector_search(search_fn)

    # Build BM25 corpus from ChromaDB
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    init_bm25_index(corpus)

    # Inject section-aware expander for complex path
    try:
        section_fetch_fn = make_section_fetch_fn(vector_store)
        rag_complex_mod.set_section_fetch_fn(section_fetch_fn)
        logger.info("v7 section-aware expander injected successfully")
    except Exception as exc:
        logger.warning("Failed to initialize section fetch for v7: %s.", exc)

    # Inject FlashRank reranker for complex path
    try:
        rerank_fn = make_rerank_fn(
            model_name=settings.RERANKING_MODEL,
            cache_dir=settings.FLASHRANK_CACHE_DIR,
        )
        rag_complex_mod.set_rerank_fn(rerank_fn)
        logger.info("v7 FlashRank reranker injected successfully")
    except Exception as exc:
        logger.warning(
            "Failed to initialize FlashRank for v7: %s. Complex path will skip reranking.",
            exc,
        )

    # Inject LLM-backed verify, rewrite, and generate functions
    if llm_provider:
        try:
            verifier_llm = get_gemini_llm(
                thinking_budget=1024, response_mime_type="application/json"
            )
            llm_verifier_mod.set_verify_fn(make_verify_fn(verifier_llm))

            rewriter_llm = get_gemini_llm(thinking_budget=1024)
            rewriter_mod.set_rewrite_fn(make_rewrite_fn(rewriter_llm))

            generator_llm = get_gemini_llm(thinking_budget=4096)
            generate_answer_mod.set_generate_fn(make_generate_fn(generator_llm))

            logger.info(
                "v7 LLM verifier, rewriter, and generator injected successfully"
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LLM for v7 verifier/rewriter/generator: %s. "
                "Using rule-based stubs.",
                exc,
            )
