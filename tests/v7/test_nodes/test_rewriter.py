"""Tests for rewriter node."""

from __future__ import annotations

import pytest

from src.v7.nodes.rewriter import rewriter


class TestRewriter:
    @pytest.mark.unit
    def test_increments_iteration(self):
        state = {
            "query": "ГОСТ 12.1.004 ограждения",
            "active_query": "ГОСТ 12.1.004 ограждения",
            "verification": {
                "rewrite_hint": "добавь раздел",
                "missing_aspects": ["числовые требования"],
            },
            "verify_iteration": 0,
        }
        result = rewriter(state)
        assert result["verify_iteration"] == 1

    @pytest.mark.unit
    def test_updates_active_query(self):
        state = {
            "query": "требования к ограждениям",
            "active_query": "требования к ограждениям",
            "verification": {"missing_aspects": ["высота"]},
            "verify_iteration": 0,
        }
        result = rewriter(state)
        assert result["active_query"] != state["active_query"]

    @pytest.mark.unit
    def test_preserves_doc_identifiers(self):
        """Doc identifiers from original query must be preserved."""
        state = {
            "query": "ГОСТ 12.1.004 требования",
            "active_query": "требования к безопасности",
            "verification": {"missing_aspects": ["конкретика"]},
            "verify_iteration": 0,
        }
        result = rewriter(state)
        assert "ГОСТ 12.1.004" in result["active_query"]

    @pytest.mark.unit
    def test_generates_new_retrieval_id(self):
        state = {
            "query": "test query",
            "active_query": "test query",
            "verification": {"missing_aspects": ["detail"]},
            "verify_iteration": 0,
        }
        result = rewriter(state)
        assert "retrieval_id" in result
