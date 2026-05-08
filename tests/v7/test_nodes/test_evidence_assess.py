"""Tests for V8 Evidence Assess — _evidence_assess(), evaluate_triage dispatcher,
and route_after_triage() under V8 flag.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.v7.nodes.evaluate_triage import (
    _evidence_assess,
    _legacy_triage,
    evaluate_triage,
    route_after_triage,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_state(
    *,
    reranker_top1: float = 0.0,
    reranker_top3_mean: float = 0.0,
    kw_overlap: float = 0.0,
    passage_count: int = 0,
    query: str = "высота ограждения лестницы",
) -> dict:
    passages = [
        {"text": f"passage {i}", "score": 0.5, "doc_id": f"d{i}"}
        for i in range(passage_count)
    ]
    metrics: dict = {
        "keyword_overlap_active": kw_overlap,
    }
    if reranker_top1 > 0.0 or reranker_top3_mean > 0.0:
        metrics["reranker_top1"] = reranker_top1
        metrics["reranker_top3_mean"] = reranker_top3_mean

    return {
        "query": query,
        "active_query": query,
        "plan": {},
        "retrieval_attempts": [
            {
                "retrieval_id": "rid1",
                "stage": "simple",
                "passages": passages,
                "top_score": 0.6,
                "attempt_plan": {},
                "metrics": metrics,
            }
        ],
    }


# ─── _evidence_assess unit tests ─────────────────────────────────────────────


class TestEvidenceAssess:
    @pytest.mark.unit
    def test_verdict_answer(self):
        """reranker_top1=0.7, kw_overlap=0.8, passage_count=15 → answer, sufficient=True."""
        state = _make_state(reranker_top1=0.7, kw_overlap=0.8, passage_count=15)
        result = _evidence_assess(state)
        assert result["sufficient"] is True
        assert result["evidence_report"]["verdict"] == "answer"
        assert "final_passages" in result

    @pytest.mark.unit
    def test_verdict_abstain(self):
        """reranker_top1=0.1, kw_overlap=0.1, passage_count=2 → abstain, sufficient=False."""
        state = _make_state(reranker_top1=0.1, kw_overlap=0.1, passage_count=2)
        result = _evidence_assess(state)
        assert result["sufficient"] is False
        assert result["evidence_report"]["verdict"] == "abstain"

    @pytest.mark.unit
    def test_verdict_improve(self):
        """reranker_top1=0.5, kw_overlap=0.4, passage_count=8 → improve, sufficient=False."""
        state = _make_state(reranker_top1=0.5, kw_overlap=0.4, passage_count=8)
        result = _evidence_assess(state)
        assert result["sufficient"] is False
        assert result["evidence_report"]["verdict"] == "improve"

    @pytest.mark.unit
    def test_evidence_report_stored_in_state(self):
        """EvidenceReport is always stored in state."""
        state = _make_state(reranker_top1=0.7, kw_overlap=0.8, passage_count=15)
        result = _evidence_assess(state)
        report = result["evidence_report"]
        assert "verdict" in report
        assert "reranker_top1" in report
        assert "coverage_estimate" in report
        assert "passage_count" in report

    @pytest.mark.unit
    def test_coverage_formula(self):
        """kw_overlap=0.5, passage_count=20 → coverage = 0.5 * min(2.0, 1.0) = 0.5."""
        state = _make_state(reranker_top1=0.0, kw_overlap=0.5, passage_count=20)
        result = _evidence_assess(state)
        report = result["evidence_report"]
        assert abs(report["coverage_estimate"] - 0.5) < 1e-9

    @pytest.mark.unit
    def test_no_passages_abstain(self):
        """No passages → abstain."""
        state = _make_state(passage_count=0)
        result = _evidence_assess(state)
        assert result["sufficient"] is False
        assert result["evidence_report"]["verdict"] == "abstain"

    @pytest.mark.unit
    def test_missing_reranker_scores_default_zero(self):
        """When metrics lack reranker_top1/top3_mean, defaults to 0.0 → likely abstain."""
        state = _make_state(kw_overlap=0.1, passage_count=2)
        # metrics do NOT contain reranker keys — only kw_overlap
        result = _evidence_assess(state)
        report = result["evidence_report"]
        assert report["reranker_top1"] == 0.0
        assert report["reranker_top3_mean"] == 0.0
        # low kw_overlap + 0 reranker → abstain
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_boundary_exactly_at_answer_threshold(self):
        """Exactly at answer thresholds (0.6, 0.6) → answer."""
        # coverage = kw_overlap * min(passage_count / 10, 1)
        # need coverage >= 0.6 → kw_overlap=0.6, passage_count=10 → coverage=0.6
        state = _make_state(reranker_top1=0.6, kw_overlap=0.6, passage_count=10)
        result = _evidence_assess(state)
        assert result["sufficient"] is True
        assert result["evidence_report"]["verdict"] == "answer"

    @pytest.mark.unit
    def test_no_retrieval_attempts(self):
        """State with no retrieval_attempts → abstain, sufficient=False."""
        state = {"query": "test", "plan": {}, "retrieval_attempts": []}
        result = _evidence_assess(state)
        assert result["sufficient"] is False
        assert result["evidence_report"]["verdict"] == "abstain"


# ─── evaluate_triage dispatcher ──────────────────────────────────────────────


class TestEvaluateTriageDispatcher:
    @pytest.mark.unit
    def test_v8_disabled_calls_legacy(self):
        """V8_ENABLE_EVIDENCE_ASSESS=False → _legacy_triage is called."""
        state = _make_state(reranker_top1=0.7, kw_overlap=0.8, passage_count=15)
        with (
            patch("src.v7.nodes.evaluate_triage.v7_config") as mock_cfg,
            patch(
                "src.v7.nodes.evaluate_triage._legacy_triage", wraps=_legacy_triage
            ) as mock_legacy,
        ):
            mock_cfg.V8_ENABLE_EVIDENCE_ASSESS = False
            evaluate_triage(state)
            mock_legacy.assert_called_once_with(state)

    @pytest.mark.unit
    def test_v8_enabled_calls_evidence_assess(self):
        """V8_ENABLE_EVIDENCE_ASSESS=True → _evidence_assess is called."""
        state = _make_state(reranker_top1=0.7, kw_overlap=0.8, passage_count=15)
        with (
            patch("src.v7.nodes.evaluate_triage.v7_config") as mock_cfg,
            patch(
                "src.v7.nodes.evaluate_triage._evidence_assess",
                wraps=_evidence_assess,
            ) as mock_assess,
        ):
            mock_cfg.V8_ENABLE_EVIDENCE_ASSESS = True
            mock_cfg.V8_EVIDENCE_ANSWER_RERANKER_TOP1 = 0.6
            mock_cfg.V8_EVIDENCE_ANSWER_COVERAGE = 0.6
            mock_cfg.V8_EVIDENCE_ABSTAIN_RERANKER_TOP1 = 0.2
            mock_cfg.V8_EVIDENCE_ABSTAIN_COVERAGE = 0.2
            evaluate_triage(state)
            mock_assess.assert_called_once_with(state)


# ─── route_after_triage with V8 evidence_report ──────────────────────────────


class TestRouteAfterTriageV8:
    @pytest.mark.unit
    def test_route_improve_to_rag_complex(self):
        """verdict=improve → sufficient=False → rag_complex (no sufficiency_details triage)."""
        state = {
            "sufficient": False,
            "query": "высота ограждения",
            "evidence_report": {"verdict": "improve"},
        }
        # no sufficiency_details → triage defaults to "clearly_bad" → rag_complex
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    def test_route_answer_non_enum_to_end(self):
        """verdict=answer (sufficient=True), non-enumeration query → end."""
        state = {
            "sufficient": True,
            "query": "высота ограждения лестницы",
            "evidence_report": {"verdict": "answer"},
        }
        assert route_after_triage(state) == "end"

    @pytest.mark.unit
    def test_route_answer_enumeration_to_rag_complex(self):
        """verdict=answer (sufficient=True), enumeration query → rag_complex."""
        state = {
            "sufficient": True,
            "query": "кто проходит обучение по программе А охраны труда",
            "evidence_report": {"verdict": "answer"},
        }
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    def test_route_after_triage_evidence_abstain(self):
        """verdict=abstain → rag_complex.

        Route: rag_complex → evaluate_complex → abstain node (when passages remain poor).
        abstain verdict has no sufficiency_details, so legacy triage branch is bypassed;
        evidence_report presence forces rag_complex directly.
        """
        state = {
            "sufficient": False,
            "query": "высота ограждения лестницы",
            "evidence_report": {"verdict": "abstain"},
        }
        assert route_after_triage(state) == "rag_complex"
