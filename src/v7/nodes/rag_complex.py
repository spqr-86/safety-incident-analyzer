"""V7 node: rag_complex — slow path with rerank + MMR."""

from __future__ import annotations

from typing import Callable, List

from src.v7.config import v7_config
from src.v7.hard_gates import compute_attempt_metrics, validate_filters
from src.v7.nodes.utils import make_retrieval_id
from src.v7.state_types import RAGState, RetrievalAttempt, RetrievalPlan

# ─── Retriever interface (injected, same as rag_simple) ──────────────────


def _default_vector_search(**kwargs) -> List[dict]:
    return []


_vector_search: Callable[..., List[dict]] = _default_vector_search


def set_vector_search(fn: Callable[..., List[dict]]) -> None:
    """Inject vector search implementation for complex retrieval."""
    global _vector_search
    _vector_search = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def rag_complex(state: RAGState) -> RAGState:
    """Slow path: higher thresholds, rerank + MMR."""
    current_plan = state.get("plan") or {}

    slow_plan: RetrievalPlan = {
        "top_k": v7_config.COMPLEX_TOP_K,
        "rerank": True,
        "timeout_ms": v7_config.COMPLEX_TIMEOUT_MS,
        "threshold": max(v7_config.COMPLEX_THRESHOLD, current_plan.get("threshold", 0)),
        "min_passages": v7_config.COMPLEX_MIN_PASSAGES,
        "min_keyword_overlap": v7_config.COMPLEX_MIN_KW_OVERLAP,
        "max_single_doc_ratio": v7_config.COMPLEX_MAX_SINGLE_DOC_RATIO,
        "borderline_threshold": v7_config.COMPLEX_BORDERLINE_THRESHOLD,
        "min_verifier_confidence": v7_config.VERIFIER_CONFIDENCE_ANCHOR,
        "require_multi_doc": current_plan.get("require_multi_doc", False),
        "mmr_lambda": current_plan.get("mmr_lambda", v7_config.MMR_LAMBDA),
    }

    active_q = state.get("active_query", state.get("query", ""))
    original_q = state.get("query", "")
    safe_filters = validate_filters(state.get("filters"))
    rid = make_retrieval_id(active_q, safe_filters)

    # Dedup
    existing = state.get("retrieval_attempts") or []
    if any(
        a.get("retrieval_id") == rid and a.get("stage") == "complex" for a in existing
    ):
        return {}

    passages = _vector_search(
        query=active_q,
        filters=safe_filters,
        top_k=12,
        rerank=True,
        mmr=True,
        mmr_lambda=slow_plan.get("mmr_lambda", 0.7),
    )
    top_score = max((p.get("score", 0.0) for p in passages), default=0.0)

    _, metrics = compute_attempt_metrics(original_q, active_q, passages, slow_plan)

    return {
        "plan": slow_plan,
        "retrieval_attempts": [
            RetrievalAttempt(
                retrieval_id=rid,
                stage="complex",
                passages=passages,
                top_score=top_score,
                attempt_plan=dict(slow_plan),
                metrics=metrics,
            )
        ],
        "status_message": f"Расширенный поиск: {len(passages)} фрагментов.",
    }
