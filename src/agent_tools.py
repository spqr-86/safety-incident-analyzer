import json
import base64
import fitz  # pymupdf
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage

from config.settings import settings
from src.llm_factory import get_vision_llm

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
def visual_proof(file_name: str, page_no: int, bbox: List[float], mode: str = "show") -> str:
    """
    Generate a visual proof (image) or analyze the visual content using AI (VLM).

    Args:
        file_name: The source file name (e.g., 'safety.pdf').
        page_no: The page number (1-based).
        bbox: The bounding box [left, top, right, bottom] from the search results.
        mode: "show" to return an image file path (default), or "analyze" to interpret the image content (e.g. for complex tables).

    Returns:
        Path to the generated image file (mode="show") or text description (mode="analyze").
    """
    try:
        # Resolve file path
        source_path = Path("source_docs") / file_name

        if not source_path.exists():
            return f"Error: File '{file_name}' not found in source_docs."

        doc = fitz.open(source_path)
        if page_no < 1 or page_no > len(doc):
            return f"Error: Page {page_no} out of range (1-{len(doc)})."

        page = doc[page_no - 1]

        l, t, r, b = bbox

        # Check if coordinates look like PDF (bottom-left) and flip if needed
        # Docling output usually [l, t, r, b] where t > b means PDF coords
        if t > b:
            height = page.rect.height
            y0 = height - t
            y1 = height - b
            # Fix order for Rect (y0 < y1)
            rect = fitz.Rect(l, y0, r, y1)
        else:
            rect = fitz.Rect(l, t, r, b)

        # Add padding
        padding = 20
        rect.x0 = max(0, rect.x0 - padding)
        rect.y0 = max(0, rect.y0 - padding)
        rect.x1 = min(page.rect.width, rect.x1 + padding)
        rect.y1 = min(page.rect.height, rect.y1 + padding)

        # Render (higher DPI for analysis)
        dpi = 200 if mode == "analyze" else 150
        pix = page.get_pixmap(clip=rect, dpi=dpi)

        # --- Mode: Analyze (VLM) ---
        if mode == "analyze":
            try:
                img_data = pix.tobytes("png")
                b64_img = base64.b64encode(img_data).decode("utf-8")
                
                vlm = get_vision_llm()
                msg = HumanMessage(content=[
                    {"type": "text", "text": "Analyze this document fragment carefully. If it is a table or diagram, transcribe its structure and data accurately. If it is text, verify the reading. Output ONLY the content description."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ])
                response = vlm.invoke([msg])
                return f"[Visual Analysis Result]\n{response.content}"
            except Exception as e:
                return f"Error in VLM analysis: {str(e)}"

        # --- Mode: Show (Default) ---
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
        return f"Error processing visual proof: {str(e)}"
