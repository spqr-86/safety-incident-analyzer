import pytest
from unittest.mock import MagicMock
import json
from langchain.docstore.document import Document
from src.file_handler import DocumentProcessor

# Correct imports based on inspection
from docling.datamodel.document import SectionHeaderItem, TextItem, ListItem
from docling.datamodel.base_models import BoundingBox
from docling_core.types.doc.document import ProvenanceItem


def test_extract_chunks_real_types():
    processor = DocumentProcessor()

    # Create real objects
    bbox = BoundingBox(l=10, b=30, r=200, t=50)
    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=[0, 10])

    # Create items with provenance
    header = SectionHeaderItem(
        text="Chapter 1",
        orig="Chapter 1",
        label="section_header",
        self_ref="#/header1",
        parent=None,
        children=[],
        prov=[prov],
    )

    text = TextItem(
        text="Content",
        orig="Content",
        label="text",
        self_ref="#/text1",
        parent=None,
        children=[],
        prov=[prov],
    )

    doc = MagicMock()
    # Mock texts() iterator to return our items
    doc.texts.return_value = iter([header, text])

    chunks = processor._process_docling_document(doc, "test.pdf")

    assert len(chunks) == 2

    # Check Header
    assert chunks[0].metadata["type"] == "header"
    assert "Chapter 1" in chunks[0].page_content
    # Check context injection
    assert "[Chapter 1]" in chunks[0].page_content

    # Check BBox
    bbox_json = chunks[0].metadata["bbox"]
    # Docling returns [l, t, r, b] where t=50, b=30 (PDF coords)
    assert json.loads(bbox_json) == [10.0, 50.0, 200.0, 30.0]

    # Check Text
    assert chunks[1].metadata["type"] == "text"
    assert "Content" in chunks[1].page_content
    assert "[Chapter 1]" in chunks[1].page_content
