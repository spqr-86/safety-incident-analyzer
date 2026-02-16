"""V7 node: router — query classification, plan creation."""

from __future__ import annotations

from src.v7.hard_gates import validate_filters
from src.v7.nodes.utils import make_retrieval_id
from src.v7.state_types import NextAfterRouter, RAGState, RetrievalPlan

# ─── Query classification (keyword-based stub, production: LLM) ──────────

COMPARISON_MARKERS = frozenset(
    {
        "сравни",
        "разница",
        "отличие",
        "коллизия",
        "приоритет",
        "что важнее",
        "противореч",
        "различия",
        "vs",
        "или лучше",
    }
)

FACTOID_MARKERS = frozenset(
    {
        "пункт",
        "п.",
        "таблица",
        "табл.",
        "значение",
        "величина",
        "минимальн",
        "максимальн",
        "не менее",
        "не более",
        "сколько",
    }
)


def _classify_query(q: str) -> tuple[bool, float]:
    """(require_multi_doc, mmr_lambda) based on query type."""
    q_lower = q.lower()
    if any(m in q_lower for m in COMPARISON_MARKERS):
        return True, 0.5
    if any(m in q_lower for m in FACTOID_MARKERS):
        return False, 0.95
    return False, 0.7


# ─── Node ─────────────────────────────────────────────────────────────────


def router(state: RAGState) -> RAGState:
    """Reads: query, filters. Writes: plan, retrieval_id, active_query, verify_iteration."""
    q = (state.get("query") or "").strip()
    filters = state.get("filters")

    if len(q) < 8:
        return {
            "clarify_message": (
                "Уточните, пожалуйста, ваш запрос. "
                "Например: какой нормативный документ или тему вы имеете в виду?"
            ),
        }

    require_multi, mmr_lambda = _classify_query(q)

    plan: RetrievalPlan = {
        "top_k": 12,
        "rerank": False,
        "timeout_ms": 250,
        "threshold": 0.45,
        "min_passages": 5,
        "min_keyword_overlap": 0.3,
        "max_single_doc_ratio": 0.8,
        "borderline_threshold": 0.25,
        "min_verifier_confidence": 0.5,
        "require_multi_doc": require_multi,
        "mmr_lambda": mmr_lambda,
    }

    return {
        "plan": plan,
        "retrieval_id": make_retrieval_id(q, validate_filters(filters)),
        "active_query": q,
        "verify_iteration": 0,
        # State cleanup
        "clarify_message": None,
        "abstain_reason": None,
        "fallback_passages": None,
        "fallback_score": None,
    }


def route_after_router(state: RAGState) -> NextAfterRouter:
    return "clarify_respond" if state.get("clarify_message") else "rag_simple"


def clarify_respond(state: RAGState) -> RAGState:
    """Pass-through. clarify_message already in state."""
    return {}
