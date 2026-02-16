"""V7 node: evaluate_triage — 3-way sufficiency gate."""

from __future__ import annotations

from typing import Any, Dict, cast

from src.v7.hard_gates import check_full_triage
from src.v7.state_types import (
    NextAfterTriage,
    RAGState,
    RetrievalPlan,
)


def evaluate_triage(state: RAGState) -> RAGState:
    """3-way gate: sufficient / borderline / clearly_bad.

    Uses check_full_triage() with plan from attempt_plan snapshot.
    Saves fallback passages when hard gates pass but soft signals escalate.
    """
    attempts = state.get("retrieval_attempts") or []
    if not attempts:
        return {"sufficient": False}

    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state["plan"])
    original_q = state.get("query", "")
    active_q = state.get("active_query", original_q)

    result = check_full_triage(original_q, active_q, last.get("passages", []), plan)

    if result["triage"] == "sufficient":
        return {
            "sufficient": True,
            "final_passages": last["passages"],
            "final_score": result["top_score"],
            "sufficiency_details": result,
        }

    update: Dict[str, Any] = {
        "sufficient": False,
        "sufficiency_details": result,
    }

    # Fallback: hard gates ok, but triage != sufficient (soft signal escalation)
    if result["sufficient"]:
        update["fallback_passages"] = last["passages"]
        update["fallback_score"] = result["top_score"]

    return cast(RAGState, update)


def route_after_triage(state: RAGState) -> NextAfterTriage:
    if state.get("sufficient"):
        return "end"
    triage = (state.get("sufficiency_details") or {}).get("triage", "clearly_bad")
    return "llm_verifier" if triage == "borderline" else "rag_complex"
