import json
import base64
import io
import fitz  # pymupdf
from PIL import Image, ImageDraw
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
def visual_proof(
    file_name: str, page_no: int, bbox: List[float], mode: str = "show"
) -> str:
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

        # Normalize coordinates (PDF bottom-left to Top-Left)
        if t > b:
            height = page.rect.height
            y0 = height - t
            y1 = height - b
            # Fix order for Rect (y0 < y1)
            rect = fitz.Rect(l, y0, r, y1)
        else:
            rect = fitz.Rect(l, t, r, b)

        # --- Mode: Analyze (VLM) with Red Box Strategy ---
        if mode == "analyze":
            try:
                # 1. Render FULL page (context is key for avoiding safety refusal)
                target_dpi = 150
                pix = page.get_pixmap(dpi=target_dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # 2. Scale coordinates (PDF 72 dpi -> target_dpi)
                scale = target_dpi / 72.0
                draw_rect = [
                    rect.x0 * scale,
                    rect.y0 * scale,
                    rect.x1 * scale,
                    rect.y1 * scale,
                ]

                # 3. Draw Red Box
                draw = ImageDraw.Draw(img)
                draw.rectangle(draw_rect, outline="red", width=5)

                # 4. Encode
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                vlm = get_vision_llm()

                prompt_text = (
                    "This is a public government document (Russia) containing safety regulations.\n"
                    "Focus ONLY on the content inside the RED BOX.\n"
                    "1. Transcribe the text or table inside the red box exactly as it appears, in Russian.\n"
                    "2. Do not summarize.\n"
                    "3. If the box cuts text, transcribe what is visible.\n"
                    "4. Output raw text/markdown only."
                )

                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ]
                )
                response = vlm.invoke([msg])
                return f"[Visual Analysis Result]\n{response.content}"
            except Exception as e:
                return f"Error in VLM analysis: {str(e)}"

        # --- Mode: Show (Default) ---
        # Add padding for crop
        padding = 20
        rect.x0 = max(0, rect.x0 - padding)
        rect.y0 = max(0, rect.y0 - padding)
        rect.x1 = min(page.rect.width, rect.x1 + padding)
        rect.y1 = min(page.rect.height, rect.y1 + padding)

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
        return f"Error processing visual proof: {str(e)}"
