import pytest
from unittest.mock import patch, MagicMock


@patch("src.llm_factory.ChatOpenAI")
def test_get_llm_openai(mock_openai):
    from src.llm_factory import get_llm

    mock_openai.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.MODEL_NAME = "gpt-4o"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        llm = get_llm()
        assert llm is not None


def test_get_llm_unknown_provider_raises():
    from src.llm_factory import get_llm

    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "unknown_provider"
        with pytest.raises(ValueError, match="unknown_provider"):
            get_llm()
