import json
import base64
import io
import fitz  # pymupdf
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Optional, Dict
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import settings
from src.llm_factory import get_vision_llm
from src.vector_store import load_vector_store

# Global retriever reference (will be set during init)
_retriever: Optional[BaseRetriever] = None
_search_call_count: int = 0
MAX_SEARCH_CALLS: int = 2


def set_global_retriever(retriever: BaseRetriever):
    global _retriever, _search_call_count
    _retriever = retriever
    _search_call_count = 0


def reset_search_counter():
    global _search_call_count
    _search_call_count = 0


def _fetch_neighboring_chunks(
    base_id: int, source: str, window: int = 2
) -> List[Document]:
    """
    Fetch chunks with IDs in range [base_id - window, base_id + window] from the same source.
    Uses direct Chroma access for reliability.
    """
    try:
        # Load vector store directly to bypass retriever wrappers (Reranker, Ensemble)
        # This ensures we have raw access to metadata filtering
        vs = load_vector_store()

        # Determine ID range
        min_id = max(0, base_id - window)
        max_id = base_id + window

        # Chroma `get` method with metadata filter
        result = vs.get(
            where={
                "$and": [
                    {"source": source},
                    {"chunk_id": {"$gte": min_id}},
                    {"chunk_id": {"$lte": max_id}},
                ]
            }
        )

        # Convert to Documents
        docs = []
        if result and result["documents"]:
            for i, text in enumerate(result["documents"]):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                docs.append(Document(page_content=text, metadata=meta))

        # Sort by chunk_id to ensure order
        docs.sort(key=lambda x: x.metadata.get("chunk_id", 0))
        return docs

    except Exception as e:
        print(f"Error fetching neighbors: {e}")
        return []


def _merge_chunks(docs: List[Document]) -> Dict:
    """
    Merge a list of sorted chunks into a single context block.
    Computes union BBox.
    """
    if not docs:
        return {}

    # Merge Content
    full_content = "\n".join([d.page_content for d in docs])

    # Base Metadata from the middle or first chunk
    base_meta = docs[0].metadata.copy()

    # Compute Union BBox
    # [min_l, min_t, max_r, max_b]
    union_bbox = None

    # We also need to handle page changes.
    # If chunks span multiple pages, visual proof might be tricky.
    # For now, let's assume we want the BBox of the *primary* page (where the hit was).
    # Or just return the bbox of the first page encountered.
    target_page = base_meta.get("page_no")

    for d in docs:
        bbox_str = d.metadata.get("bbox")
        page = d.metadata.get("page_no")

        if bbox_str and page == target_page:
            try:
                bbox = json.loads(bbox_str)
                if union_bbox is None:
                    union_bbox = list(bbox)
                else:
                    union_bbox[0] = min(union_bbox[0], bbox[0])
                    union_bbox[1] = min(union_bbox[1], bbox[1])
                    union_bbox[2] = max(union_bbox[2], bbox[2])
                    union_bbox[3] = max(union_bbox[3], bbox[3])
            except:
                pass

    return {
        "content": full_content,
        "bbox": union_bbox,
        "page_no": target_page,
        "source": base_meta.get("source"),
        "chunk_ids": [d.metadata.get("chunk_id") for d in docs],
    }


@tool
def search_documents(query: str) -> str:
    """
    Search for information in the safety regulations.
    Returns RELEVANT text chunks with their ID, Source File, Page Number, and Bounding Box (bbox).

    The tool automatically expands context (fetches neighboring paragraphs) to provide complete information.

    Use this tool to find the information, then use the visual_proof tool with the extracted details.
    """
    global _retriever, _search_call_count
    if not _retriever:
        return "Error: Retriever not initialized."

    _search_call_count += 1
    if _search_call_count > MAX_SEARCH_CALLS:
        return (
            f"Лимит поисков достигнут ({MAX_SEARCH_CALLS} из {MAX_SEARCH_CALLS}). "
            "Сформулируй ответ на основе уже найденных данных."
        )

    # 1. Initial Retrieval
    initial_docs = _retriever.invoke(query)

    if not initial_docs:
        return "No relevant documents found."

    # 2. Smart Context Extension & Deduplication
    # We want to group overlapping ranges.
    # Map: (source, chunk_id) -> Document

    # Identify unique hits (source, chunk_id)
    hits = set()
    for doc in initial_docs:
        meta = doc.metadata
        if "chunk_id" in meta and "source" in meta:
            hits.add((meta["source"], meta["chunk_id"]))

    # If we can't find chunk_id, fall back to simple return
    if not hits:
        # Fallback logic identical to old version
        results = []
        for i, doc in enumerate(initial_docs):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page_no", "N/A")
            bbox = meta.get("bbox", "N/A")
            content = doc.page_content.replace("\n", " ")[:500]
            results.append(
                f"[ID: {i}] File: {source} | Page: {page} | BBox: {bbox}\nContent: {content}..."
            )
        return "\n\n".join(results)

    # Group hits by source
    hits_by_source = {}
    for source, cid in hits:
        if source not in hits_by_source:
            hits_by_source[source] = []
        hits_by_source[source].append(cid)

    final_blocks = []

    # Process each source
    for source, cids in hits_by_source.items():
        # Sort IDs
        cids.sort()

        # Merge ranges: [10, 11, 12, 15] with window 2 -> [8..14], [13..17] -> overlaps -> [8..17]
        # Simplified: Just fetch neighbors for each, keep track of fetched IDs

        # Let's fetch ranges
        ranges = []
        if not cids:
            continue

        curr_start = cids[0] - 2
        curr_end = cids[0] + 2

        for cid in cids[1:]:
            start = cid - 2
            end = cid + 2

            if start <= curr_end + 1:  # Overlap or adjacent
                curr_end = max(curr_end, end)
            else:
                ranges.append((curr_start, curr_end))
                curr_start = start
                curr_end = end
        ranges.append((curr_start, curr_end))

        # Fetch and Merge
        for start, end in ranges:
            # We need to fetch chunks [start, end]
            # Since we can't easily do range query without direct DB access in this scope,
            # We will use the helper _fetch_neighboring_chunks but logic modified to take range.

            # Re-use helper logic but for explicit range
            try:
                # Direct access to vector store
                from src.vector_store import load_vector_store

                vs = load_vector_store()

                result = vs.get(
                    where={
                        "$and": [
                            {"source": source},
                            {"chunk_id": {"$gte": start}},
                            {"chunk_id": {"$lte": end}},
                        ]
                    }
                )

                if result and result["documents"]:
                    range_docs = []
                    for k, txt in enumerate(result["documents"]):
                        m = result["metadatas"][k]
                        range_docs.append(Document(page_content=txt, metadata=m))

                    range_docs.sort(key=lambda x: x.metadata.get("chunk_id", 0))

                    # Merge
                    merged = _merge_chunks(range_docs)
                    if merged:
                        final_blocks.append(merged)
            except Exception as e:
                print(f"Error processing range {start}-{end} for {source}: {e}")
                continue

    # 3. Format Output
    output = []
    for i, block in enumerate(final_blocks):
        src = block["source"]
        pg = block["page_no"]
        bbox = block["bbox"]
        txt = block["content"]
        # Limit length but keep it generous since it's "Smart Context"
        txt_preview = txt[:2000]

        output.append(
            f"[Result {i}] File: {src} | Page: {pg} | BBox: {bbox}\n"
            f"Extended Context:\n{txt_preview}\n"
            f"(IDs: {block['chunk_ids']})"
        )

    return "\n\n".join(output)


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

        # --- Mode: Show (Default) with Visual Zoom (Padding) ---
        # Add generous padding for context (50px)
        padding = 50
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
