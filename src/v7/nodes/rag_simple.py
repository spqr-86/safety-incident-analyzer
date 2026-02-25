"""V7 node: rag_simple — fast hybrid retrieval (vector + BM25 → RRF)."""

from __future__ import annotations

from typing import Callable, List

from src.v7.hard_gates import compute_attempt_metrics, validate_filters
from src.v7.nlp_core import bm25_search, rrf_merge
from src.v7.state_types import RAGState, RetrievalAttempt

# ─── Retriever interface (injected at graph build time) ──────────────────


# Default: stub that returns empty. Production: Chroma/Qdrant vector search.
def _default_vector_search(**kwargs) -> List[dict]:
    return []


_vector_search: Callable[..., List[dict]] = _default_vector_search


def set_vector_search(fn: Callable[..., List[dict]]) -> None:
    """Inject vector search implementation. Call once at startup."""
    global _vector_search
    _vector_search = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def rag_simple(state: RAGState) -> RAGState:
    """Fast hybrid retrieval: vector + BM25 → RRF merge."""
    plan = state["plan"]
    rid = state["retrieval_id"]
    active_q = state.get("active_query", state["query"])
    original_q = state.get("query", "")
    safe_filters = validate_filters(state.get("filters"))

    # Dedup: skip if already executed for this retrieval_id + stage
    existing = state.get("retrieval_attempts") or []
    if any(
        a.get("retrieval_id") == rid and a.get("stage") == "simple" for a in existing
    ):
        return {}

    # Vector search
    vector_results = _vector_search(
        query=active_q,
        filters=safe_filters,
        top_k=plan["top_k"],
    )

    # BM25 full-text search
    bm25_results = bm25_search(
        query=active_q,
        filters=safe_filters,
        top_k=plan["top_k"],
    )

    # RRF merge
    passages = rrf_merge(vector_results, bm25_results, top_k=plan["top_k"])
    if not passages:
        passages = sorted(
            vector_results, key=lambda x: x.get("score", 0.0), reverse=True
        )

    top_score = max((p.get("score", 0.0) for p in passages), default=0.0)

    _, metrics = compute_attempt_metrics(original_q, active_q, passages, plan)
    metrics["retrieval_type"] = "hybrid_rrf"

    return {
        "retrieval_attempts": [
            RetrievalAttempt(
                retrieval_id=rid,
                stage="simple",
                passages=passages,
                top_score=top_score,
                attempt_plan=dict(plan),
                metrics=metrics,
            )
        ],
        "status_message": f"Найдено {len(passages)} фрагментов (hybrid search).",
    }
