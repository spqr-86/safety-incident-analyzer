"""Тесты роутинга _route_after_verify в MultiAgentRAGWorkflow.

Регрессия: поле escalated_from_simple отсутствовало в RAGState TypedDict —
LangGraph его не сохранял → ревизия после escalation шла в rag_simple
и теряла decomposition. Здесь фиксируем корректный путь revision-петли.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.multiagent_rag import MultiAgentRAGWorkflow, RAGState
from src.types import VerifyStatus


def _make_state(**overrides) -> RAGState:
    base: dict = {
        "query": "тест",
        "verify_status": VerifyStatus.NEEDS_REVISION,
        "revision_count": 0,
        "escalated_from_simple": False,
        "subquestions": [],
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


@pytest.fixture
def workflow():
    with patch.object(MultiAgentRAGWorkflow, "__init__", lambda self, *a, **k: None):
        wf = MultiAgentRAGWorkflow()
    return wf


def test_route_after_verify_approved_goes_to_format_final(workflow):
    state = _make_state(verify_status=VerifyStatus.APPROVED)
    assert workflow._route_after_verify(state) == "format_final"


def test_route_after_verify_revision_after_escalation_goes_to_complex(workflow):
    """Главная регрессия: после escalation ревизия должна вернуться в rag_complex."""
    state = _make_state(escalated_from_simple=True)
    assert workflow._route_after_verify(state) == "rag_complex"


def test_route_after_verify_revision_without_escalation_goes_to_simple(workflow):
    state = _make_state(escalated_from_simple=False)
    assert workflow._route_after_verify(state) == "rag_simple"


def test_route_after_verify_max_revisions_goes_to_format_final(workflow):
    from config.settings import settings

    state = _make_state(
        revision_count=settings.MAX_REVISIONS + 1, escalated_from_simple=True
    )
    assert workflow._route_after_verify(state) == "format_final"
