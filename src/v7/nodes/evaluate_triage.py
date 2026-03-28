"""V7 node: evaluate_triage — 3-way sufficiency gate."""

from __future__ import annotations

import re
from typing import Any, Dict, cast

from src.v7.hard_gates import check_full_triage
from src.v7.state_types import (
    NextAfterTriage,
    RAGState,
    RetrievalPlan,
)

# Паттерны перечислительных вопросов — требуют полного покрытия всех категорий/условий.
# Для таких запросов rag_simple может вернуть неполный ответ даже при высоком top_score.
_ENUMERATION_PATTERNS = [
    r"\bкто\s+проходит\b",
    r"\bкто\s+обязан\b",
    r"\bкакие\s+категори[яи]\b",
    r"\bв\s+каких\s+случаях\b",
    r"\bкогда\s+не\s+требуется\b",
    r"\bкому\s+не\s+требуется\b",
    r"\bкто\s+освобождается\b",
    r"\bперечислите\b",
    r"\bкакие\s+работники\b",
    r"\bкаким\s+работникам\b",
]


# Паттерны перекрёстных ссылок в нормативных документах.
# Если retrieved чанки содержат много таких маркеров — ответ, вероятно, рассеян
# по соседним пунктам и требует расширенного поиска.
_CROSSREF_PATTERNS = [
    r"\bпункт[а-я]*\s+\d+",
    r"\bподпункт[а-я]*\s+\d+",
    r"\bза\s+исключением\b",
    r"\bв\s+соответствии\s+с\b",
    r"\bуказанн[а-я]+\s+в\b",
    r"\bсогласно\s+пункт",
    r"\bсм\.\s+пункт",
    r"\bприложени[яе]\s+\d+",
]

_CROSSREF_ESCALATION_THRESHOLD = 3  # >= N hits в топ-чанках → escalate


def _count_crossref_hits(passages: list[dict]) -> int:
    """Считает суммарное число crossref-паттернов в топ-5 passages."""
    total = 0
    for p in passages[:5]:
        text = p.get("text", "").lower()
        for pattern in _CROSSREF_PATTERNS:
            if re.search(pattern, text):
                total += 1
    return total


def _has_enumeration_intent(query: str) -> bool:
    """True если запрос требует полного перечисления категорий/условий.

    Такие запросы направляются в rag_complex даже при sufficient simple-triage,
    потому что ответ часто рассеян по нескольким пунктам документа.
    """
    q = query.lower()
    return any(re.search(p, q) for p in _ENUMERATION_PATTERNS)


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
        # Crossref escalation: many cross-references in retrieved chunks indicate
        # the answer is distributed across multiple document sections.
        # Save as fallback and escalate to rag_complex for fuller coverage.
        passages = last.get("passages", [])
        crossref_hits = _count_crossref_hits(passages)
        if crossref_hits >= _CROSSREF_ESCALATION_THRESHOLD:
            return {
                "sufficient": False,
                "sufficiency_details": result,
                "fallback_passages": passages,
                "fallback_score": result["top_score"],
            }
        return {
            "sufficient": True,
            "final_passages": passages,
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
        # Enumeration queries require complete coverage across multiple document sections.
        # Force rag_complex even when simple-triage scores are sufficient.
        if _has_enumeration_intent(state.get("query", "")):
            return "rag_complex"
        return "end"
    triage = (state.get("sufficiency_details") or {}).get("triage", "clearly_bad")
    return "llm_verifier" if triage == "borderline" else "rag_complex"
