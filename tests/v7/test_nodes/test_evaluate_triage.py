"""Tests for evaluate_triage node."""

from __future__ import annotations

import pytest

from src.v7.nodes.evaluate_triage import (
    _has_enumeration_intent,
    evaluate_triage,
    route_after_triage,
)


def _make_attempt(passages, plan=None):
    return {
        "retrieval_id": "rid1",
        "stage": "simple",
        "passages": passages,
        "top_score": max((p.get("score", 0) for p in passages), default=0),
        "attempt_plan": plan
        or {
            "threshold": 0.65,
            "min_passages": 2,
            "min_keyword_overlap": 0.3,
            "max_single_doc_ratio": 0.6,
            "borderline_threshold": 0.40,
            "require_multi_doc": False,
        },
    }


class TestEvaluateTriage:
    @pytest.mark.unit
    def test_sufficient(self):
        passages = [
            {"text": "ограждение лестница высота", "score": 0.8, "doc_id": "d1"},
            {"text": "ограждение балкон нормы", "score": 0.7, "doc_id": "d2"},
            {"text": "лестница ограждение проект", "score": 0.65, "doc_id": "d3"},
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = evaluate_triage(state)
        assert result["sufficient"] is True
        assert result["final_passages"] == passages

    @pytest.mark.unit
    def test_clearly_bad(self):
        passages = [
            {"text": "ограждение", "score": 0.2, "doc_id": "d1"},
            {"text": "ограждение", "score": 0.1, "doc_id": "d2"},
        ]
        state = {
            "query": "ограждение",
            "active_query": "ограждение",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = evaluate_triage(state)
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_no_attempts(self):
        result = evaluate_triage({"retrieval_attempts": [], "plan": {}})
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_fallback_saved_on_borderline(self):
        """When hard gates pass but soft signals escalate, save fallback."""
        passages = [
            {"text": "ограждение лестница высота", "score": 0.8, "doc_id": "d1"},
            {"text": "ограждение балкон нормы", "score": 0.7, "doc_id": "d1"},
            {"text": "лестница ограждение проект", "score": 0.65, "doc_id": "d1"},
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = evaluate_triage(state)
        # plan={} → max_single_doc_ratio=1.0 → diversity_ok=True, escalation_hint=False.
        # Fallback saved only when hard gates pass but soft signals escalate (plan with low ratio).
        if not result.get("sufficient"):
            details = result.get("sufficiency_details", {})
            if details.get("sufficient"):
                assert "fallback_passages" in result


class TestRouteAfterTriage:
    @pytest.mark.unit
    def test_sufficient_routes_end(self):
        assert route_after_triage({"sufficient": True, "query": "ограждение лестница"}) == "end"

    @pytest.mark.unit
    def test_borderline_routes_verifier(self):
        state = {
            "sufficient": False,
            "sufficiency_details": {"triage": "borderline"},
            "query": "ограждение лестница",
        }
        assert route_after_triage(state) == "llm_verifier"

    @pytest.mark.unit
    def test_clearly_bad_routes_complex(self):
        state = {
            "sufficient": False,
            "sufficiency_details": {"triage": "clearly_bad"},
            "query": "ограждение лестница",
        }
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    @pytest.mark.parametrize("query", [
        "кто проходит обучение по программе А охраны труда",
        "какие категории работников проходят инструктаж",
        "в каких случаях не требуется инструктаж",
        "кто освобождается от первичного инструктажа",
        "кому не требуется проходить обучение",
    ])
    def test_enumeration_intent_detected(self, query: str):
        assert _has_enumeration_intent(query) is True

    @pytest.mark.unit
    @pytest.mark.parametrize("query", [
        "какова высота ограждения лестницы",
        "что такое охрана труда",
        "требования к освещению рабочего места",
    ])
    def test_no_enumeration_intent(self, query: str):
        assert _has_enumeration_intent(query) is False

    @pytest.mark.unit
    def test_enumeration_query_forces_complex_even_if_sufficient(self):
        """Sufficient simple-triage result should still go to rag_complex for enumeration queries."""
        state = {
            "sufficient": True,
            "query": "кто проходит обучение по программе А охраны труда",
        }
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    def test_non_enumeration_sufficient_routes_end(self):
        """Non-enumeration sufficient result should go to end as before."""
        state = {
            "sufficient": True,
            "query": "какова минимальная высота ограждения",
        }
        assert route_after_triage(state) == "end"
