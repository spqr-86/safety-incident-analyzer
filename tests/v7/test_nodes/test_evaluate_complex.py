"""Tests for evaluate_complex node."""

from __future__ import annotations

import pytest

from src.v7.nodes.evaluate_complex import evaluate_complex, route_after_eval_complex

PLAN = {
    "threshold": 0.5,
    "min_passages": 2,
    "min_keyword_overlap": 0.2,
    "max_single_doc_ratio": 1.0,
}


class TestEvaluateComplex:
    @pytest.mark.unit
    def test_sufficient_from_merged(self):
        attempts = [
            {
                "passages": [
                    {
                        "chunk_id": "c1",
                        "text": "ограждение лестница",
                        "score": 0.7,
                        "doc_id": "d1",
                    },
                    {
                        "chunk_id": "c2",
                        "text": "ограждение балкон",
                        "score": 0.6,
                        "doc_id": "d2",
                    },
                ],
                "attempt_plan": PLAN,
            },
            {
                "passages": [
                    {
                        "chunk_id": "c3",
                        "text": "лестница нормы ограждение",
                        "score": 0.55,
                        "doc_id": "d3",
                    },
                ],
                "attempt_plan": PLAN,
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": attempts,
        }
        result = evaluate_complex(state)
        assert result["sufficient"] is True
        assert len(result["final_passages"]) == 3

    @pytest.mark.unit
    def test_fallback_used(self):
        """When merged and last fail, fallback passages should be used."""
        attempts = [
            {
                "passages": [
                    {
                        "chunk_id": "c1",
                        "text": "нерелевантное",
                        "score": 0.1,
                        "doc_id": "d1",
                    },
                ],
                "attempt_plan": PLAN,
            },
        ]
        fallback = [
            {
                "chunk_id": "f1",
                "text": "ограждение лестница",
                "score": 0.7,
                "doc_id": "d1",
            },
            {
                "chunk_id": "f2",
                "text": "ограждение балкон",
                "score": 0.6,
                "doc_id": "d2",
            },
        ]
        state = {
            "query": "ограждение",
            "active_query": "ограждение",
            "retrieval_attempts": attempts,
            "fallback_passages": fallback,
            "fallback_score": 0.7,
        }
        result = evaluate_complex(state)
        assert result["sufficient"] is True
        assert result["final_passages"] == fallback

    @pytest.mark.unit
    def test_full_failure(self):
        attempts = [
            {
                "passages": [{"text": "нерелевантное", "score": 0.1, "doc_id": "d1"}],
                "attempt_plan": PLAN,
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": attempts,
        }
        result = evaluate_complex(state)
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_no_attempts(self):
        result = evaluate_complex({"retrieval_attempts": []})
        assert result["sufficient"] is False


class TestRouteAfterEvalComplex:
    @pytest.mark.unit
    def test_sufficient_end(self):
        assert route_after_eval_complex({"sufficient": True}) == "end"

    @pytest.mark.unit
    def test_fail_abstain(self):
        assert route_after_eval_complex({"sufficient": False}) == "abstain"
