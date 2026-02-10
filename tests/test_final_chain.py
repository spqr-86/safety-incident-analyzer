from unittest.mock import MagicMock, patch


@patch("src.final_chain.Ranker")
@patch("src.final_chain.get_llm")
@patch("src.final_chain.ApplicabilityRetriever")
@patch("src.final_chain.FlashrankRerank")
@patch("src.final_chain.ContextualCompressionRetriever")
def test_build_reranked_retriever_creates_retriever(
    mock_ccr, mock_flashrank, mock_app_retriever, mock_get_llm, mock_ranker
):
    from src.final_chain import build_reranked_retriever

    mock_vs = MagicMock()
    mock_bm25 = MagicMock()
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    mock_ensemble = MagicMock()
    mock_app_retriever.return_value = mock_ensemble

    mock_compressor = MagicMock()
    mock_flashrank.return_value = mock_compressor

    mock_retriever = MagicMock()
    mock_ccr.return_value = mock_retriever

    retriever = build_reranked_retriever(mock_vs, mock_bm25, mock_llm)
    assert retriever is mock_retriever

    mock_app_retriever.assert_called_once()
    mock_flashrank.assert_called_once()
    mock_ccr.assert_called_once()


@patch("src.final_chain.Ranker")
@patch("src.final_chain.get_llm")
@patch("src.final_chain.ApplicabilityRetriever")
@patch("src.final_chain.FlashrankRerank")
@patch("src.final_chain.ContextualCompressionRetriever")
def test_build_reranked_retriever_respects_query_expansion_flag(
    mock_ccr, mock_flashrank, mock_app_retriever, mock_get_llm, mock_ranker
):
    from src.final_chain import build_reranked_retriever

    mock_vs = MagicMock()
    mock_bm25 = MagicMock()
    mock_llm = MagicMock()

    mock_ensemble = MagicMock()
    mock_app_retriever.return_value = mock_ensemble

    mock_compressor = MagicMock()
    mock_flashrank.return_value = mock_compressor

    mock_retriever = MagicMock()
    mock_ccr.return_value = mock_retriever

    r1 = build_reranked_retriever(mock_vs, mock_bm25, mock_llm, query_expansion=True)
    r2 = build_reranked_retriever(mock_vs, mock_bm25, mock_llm, query_expansion=False)

    assert r1 is mock_retriever
    assert r2 is mock_retriever

    assert mock_app_retriever.call_count == 2

    # Check arguments
    assert mock_app_retriever.call_args_list[0].kwargs["query_expansion"] is True
    assert mock_app_retriever.call_args_list[1].kwargs["query_expansion"] is False
