"""Tests for rag_simple node."""

from __future__ import annotations

import pytest

from src.v7.nodes.rag_simple import rag_simple


def _make_state(active_query="ограждение лестница", **overrides):
    state = {
        "query": active_query,
        "active_query": active_query,
        "plan": {
            "top_k": 5,
            "rerank": False,
            "timeout_ms": 100,
            "threshold": 0.45,
            "min_passages": 2,
            "min_keyword_overlap": 0.2,
        },
        "retrieval_id": "test_rid_001",
        "retrieval_attempts": [],
    }
    state.update(overrides)
    return state


class TestRagSimple:
    @pytest.mark.unit
    def test_returns_attempt(self):
        """With default stub (empty results), still returns attempt structure."""
        result = rag_simple(_make_state())
        attempts = result.get("retrieval_attempts", [])
        assert len(attempts) == 1
        assert attempts[0]["stage"] == "simple"
        assert attempts[0]["retrieval_id"] == "test_rid_001"
        assert "status_message" in result

    @pytest.mark.unit
    def test_dedup_skips_existing(self):
        """Skip if same retrieval_id + stage already exists."""
        existing_attempt = {
            "retrieval_id": "test_rid_001",
            "stage": "simple",
            "passages": [],
            "top_score": 0.0,
        }
        state = _make_state(retrieval_attempts=[existing_attempt])
        result = rag_simple(state)
        assert result == {}

    @pytest.mark.unit
    def test_attempt_has_metrics(self):
        result = rag_simple(_make_state())
        attempt = result["retrieval_attempts"][0]
        assert "metrics" in attempt
        assert "attempt_plan" in attempt
        assert attempt["metrics"]["retrieval_type"] == "hybrid_rrf"
