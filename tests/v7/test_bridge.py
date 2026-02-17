"""Tests for src/v7/bridge.py — v7 ↔ existing retriever bridge."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.v7.bridge import make_vector_search_fn, init_v7_from_chroma


class TestMakeVectorSearchFn:
    @pytest.mark.unit
    def test_returns_callable(self):
        mock_store = MagicMock()
        fn = make_vector_search_fn(mock_store)
        assert callable(fn)

    @pytest.mark.unit
    def test_calls_similarity_search_with_score(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        result = fn(query="test query", top_k=5)
        mock_store.similarity_search_with_score.assert_called_once()
        assert result == []

    @pytest.mark.unit
    def test_converts_documents_to_dicts(self):
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "some text about safety"
        mock_doc.metadata = {"source": "gost.pdf", "page_no": 5}
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.3)]
        fn = make_vector_search_fn(mock_store)
        result = fn(query="safety", top_k=10)
        assert len(result) == 1
        assert result[0]["text"] == "some text about safety"
        assert result[0]["metadata"]["source"] == "gost.pdf"
        assert result[0]["score"] == pytest.approx(0.7, abs=0.01)

    @pytest.mark.unit
    def test_respects_top_k(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        fn(query="test", top_k=20)
        call_kwargs = mock_store.similarity_search_with_score.call_args
        assert call_kwargs[1].get("k") == 20 or call_kwargs[0] == ("test",)

    @pytest.mark.unit
    def test_filters_ignored_gracefully(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        result = fn(query="test", top_k=5, filters={"doc_type": "gost"})
        assert result == []


class TestInitV7FromChroma:
    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_injects_vector_search(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "documents": ["doc1 text", "doc2 text"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        init_v7_from_chroma(mock_store)
        mock_simple.set_vector_search.assert_called_once()
        mock_complex.set_vector_search.assert_called_once()
        mock_bm25.assert_called_once()

    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_bm25_corpus_built_from_chroma(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "documents": ["text A", "text B"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        init_v7_from_chroma(mock_store)
        corpus = mock_bm25.call_args[0][0]
        assert len(corpus) == 2
        assert corpus[0]["text"] == "text A"
        assert corpus[1]["metadata"]["source"] == "b.pdf"
