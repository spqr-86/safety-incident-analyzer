import json
import fitz  # pymupdf
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever

from config.settings import settings

# Global retriever reference (will be set during init)
_retriever: Optional[BaseRetriever] = None


def set_global_retriever(retriever: BaseRetriever):
    global _retriever
    _retriever = retriever


@tool
def search_documents(query: str) -> str:
    """
    Search for information in the safety regulations.
    Returns relevant text chunks with their ID, Source File, Page Number, and Bounding Box (bbox).

    Use this tool to find the information, then use the visual_proof tool with the extracted details.
    """
    global _retriever
    if not _retriever:
        return "Error: Retriever not initialized."

    docs = _retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    results = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page_no", "N/A")
        bbox = meta.get("bbox", "N/A")
        content = doc.page_content.replace("\n", " ")[:500]  # Limit content length

        results.append(
            f"[ID: {i}] File: {source} | Page: {page} | BBox: {bbox}\n"
            f"Content: {content}..."
        )

    return "\n\n".join(results)


@tool
def visual_proof(file_name: str, page_no: int, bbox: List[float]) -> str:
    """
    Generate a visual proof (image) from a document using the provided coordinates.

    Args:
        file_name: The source file name (e.g., 'safety.pdf').
        page_no: The page number (1-based).
        bbox: The bounding box [left, top, right, bottom] from the search results.

    Returns:
        Path to the generated image file.
    """
    try:
        # Resolve file path
        # Assuming source_docs are in a known location or we scan for them
        # For simplicity, we assume they are in 'source_docs/' relative to project root
        # OR we might need a mapping. Let's assume flat structure in source_docs/
        source_path = Path("source_docs") / file_name

        if not source_path.exists():
            return f"Error: File '{file_name}' not found in source_docs."

        doc = fitz.open(source_path)
        if page_no < 1 or page_no > len(doc):
            return f"Error: Page {page_no} out of range (1-{len(doc)})."

        page = doc[page_no - 1]

        # BBox format: [l, t, r, b]
        # PyMuPDF expects Rect(x0, y0, x1, y1)
        # We need to handle potential coordinate system mismatch if necessary
        # But earlier test showed [l, t, r, b] where t > b (PDF coords).
        # PyMuPDF uses Top-Left origin (y grows down).
        # PDF uses Bottom-Left origin (y grows up).
        # So we might need to flip Y.
        # Page height is page.rect.height

        l, t, r, b = bbox

        # Check if coordinates look like PDF (bottom-left)
        # If t > b, it's likely PDF coordinates.
        if t > b:
            # Convert to PyMuPDF (Top-Left)
            # y_new = height - y_old
            height = page.rect.height

            # PDF Top (t) is distance from bottom.
            # PyMuPDF Top (y0) is distance from top.
            # y0 = height - t
            # y1 = height - b

            # Wait, PDF convention:
            # (0,0) is bottom-left.
            # t=50 means 50 units from bottom?
            # Or does Docling return (l, b, r, t)?
            # My test showed [10, 50, 200, 30]. t=50, b=30.
            # In PDF, y=50 is higher than y=30.
            # So top edge is y=50, bottom edge is y=30.
            # To convert to Top-Left:
            # y0 (top edge) = PageHeight - 50
            # y1 (bottom edge) = PageHeight - 30

            y0 = height - t
            y1 = height - b

            # Fix order for Rect (y0 < y1)
            rect = fitz.Rect(l, y0, r, y1)
        else:
            # Assumed valid Top-Left coordinates
            rect = fitz.Rect(l, t, r, b)

        # Add padding
        padding = 20
        rect.x0 = max(0, rect.x0 - padding)
        rect.y0 = max(0, rect.y0 - padding)
        rect.x1 = min(page.rect.width, rect.x1 + padding)
        rect.y1 = min(page.rect.height, rect.y1 + padding)

        # Render
        pix = page.get_pixmap(clip=rect, dpi=150)

        # Save
        output_dir = Path("static/visuals")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Hash params to make filename unique but deterministic
        import hashlib

        h = hashlib.md5(f"{file_name}_{page_no}_{bbox}".encode()).hexdigest()[:8]
        output_filename = f"proof_{h}.png"
        output_path = output_dir / output_filename

        pix.save(output_path)

        return str(output_path)

    except Exception as e:
        return f"Error generating visual proof: {str(e)}"
