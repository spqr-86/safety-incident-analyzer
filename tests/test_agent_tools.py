import pytest
from unittest.mock import MagicMock, patch
from src.agent_tools import visual_proof

@patch("src.agent_tools.fitz.open")
@patch("src.agent_tools.get_vision_llm")
@patch("src.agent_tools.Path.exists", return_value=True) # Mock file exists
def test_visual_proof_analyze_mode(mock_exists, mock_get_llm, mock_fitz_open):
    # Setup Mock PDF
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.rect.width = 100
    mock_page.rect.height = 100
    # Mock render result
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"fake_image_data"
    
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.__len__.return_value = 10 # Ensure page 1 is valid
    mock_fitz_open.return_value = mock_doc
    
    # Setup Mock LLM
    mock_vlm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is a table description."
    mock_vlm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_vlm
    
    # Execute
    result = visual_proof.invoke({
        "file_name": "test.pdf", 
        "page_no": 1, 
        "bbox": [10, 10, 50, 50], 
        "mode": "analyze"
    })
    
    # Verify
    assert "[Visual Analysis Result]" in result
    assert "This is a table description" in result
    
    mock_vlm.invoke.assert_called_once()

@patch("src.agent_tools.fitz.open")
def test_visual_proof_show_mode(mock_fitz_open):
    # Setup Mock PDF
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.rect.width = 100 # Fix: Set dimensions
    mock_page.rect.height = 100 # Fix: Set dimensions
    
    mock_pix = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.__len__.return_value = 10
    mock_fitz_open.return_value = mock_doc
    
    with patch("src.agent_tools.Path.mkdir"):
        with patch("src.agent_tools.Path.exists", return_value=True):
             # Execute
             result = visual_proof.invoke({
                 "file_name": "test.pdf", 
                 "page_no": 1, 
                 "bbox": [10, 10, 50, 50], 
                 "mode": "show"
             })
             
             # Verify it returns a path string
             assert "static/visuals/proof_" in result
             assert result.endswith(".png")
             
             # Verify save called
             mock_pix.save.assert_called()
