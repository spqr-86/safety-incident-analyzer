"""Tests for nodes/utils.py."""

from __future__ import annotations

import pytest

from src.v7.nodes.utils import extract_doc_identifiers, make_retrieval_id


class TestMakeRetrievalId:
    @pytest.mark.unit
    def test_deterministic(self):
        id1 = make_retrieval_id("test query", {"doc_type": "ГОСТ"})
        id2 = make_retrieval_id("test query", {"doc_type": "ГОСТ"})
        assert id1 == id2

    @pytest.mark.unit
    def test_different_queries(self):
        id1 = make_retrieval_id("query A")
        id2 = make_retrieval_id("query B")
        assert id1 != id2

    @pytest.mark.unit
    def test_length(self):
        rid = make_retrieval_id("test")
        assert len(rid) == 16


class TestExtractDocIdentifiers:
    @pytest.mark.unit
    def test_gost(self):
        ids = extract_doc_identifiers("ГОСТ 12.1.004-91")
        assert any("ГОСТ" in i for i in ids)

    @pytest.mark.unit
    def test_sp(self):
        ids = extract_doc_identifiers("СП 1.13130")
        assert any("СП" in i for i in ids)

    @pytest.mark.unit
    def test_multiple(self):
        ids = extract_doc_identifiers("ГОСТ 12.1.004 и СП 1.13130")
        assert len(ids) >= 2

    @pytest.mark.unit
    def test_no_identifiers(self):
        ids = extract_doc_identifiers("просто текст без номеров")
        assert len(ids) == 0
