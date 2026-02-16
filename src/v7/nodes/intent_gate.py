"""V7 node: intent_gate — noise/domain classification."""

from __future__ import annotations

from src.v7.state_types import NextAfterIntent, RAGState

NOISE_QUERIES = frozenset({"привет", "здравствуй", "как дела", "hello", "hi", "hey"})


def intent_gate(state: RAGState) -> RAGState:
    """Reads: query. Writes: intent."""
    q = (state.get("query") or "").strip()
    if len(q) < 3 or q.lower() in NOISE_QUERIES:
        return {"intent": "noise"}
    return {"intent": "domain"}


def route_by_intent(state: RAGState) -> NextAfterIntent:
    return "end" if state["intent"] == "noise" else "router"
