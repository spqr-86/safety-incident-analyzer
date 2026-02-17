"""Bridge: adapt existing ChromaDB vector store for v7 pipeline.

Responsibilities:
1. Wrap ChromaDB's similarity_search_with_score -> v7 dict format
2. Build BM25 corpus from ChromaDB docs
3. Inject search functions into rag_simple / rag_complex nodes
"""

from __future__ import annotations

from typing import Callable, List

from src.v7.nlp_core import init_bm25_index
from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes import rag_complex as rag_complex_mod


def make_vector_search_fn(vector_store) -> Callable[..., List[dict]]:
    """Create a v7-compatible vector search function from ChromaDB store.

    v7 interface: fn(query, filters=None, top_k=12, **kwargs) -> list[dict]
    Each dict has: text, metadata, score.
    """

    def _search(
        query: str,
        filters: dict | None = None,
        top_k: int = 12,
        **kwargs,
    ) -> List[dict]:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, distance in docs_and_scores:
            similarity = max(0.0, 1.0 - distance)
            results.append(
                {
                    "text": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": round(similarity, 4),
                }
            )
        return results

    return _search


def init_v7_from_chroma(vector_store) -> None:
    """Initialize v7 pipeline from existing ChromaDB vector store.

    1. Creates vector search wrapper
    2. Injects it into rag_simple and rag_complex nodes
    3. Builds BM25 index from full corpus
    """
    search_fn = make_vector_search_fn(vector_store)
    rag_simple_mod.set_vector_search(search_fn)
    rag_complex_mod.set_vector_search(search_fn)

    # Build BM25 corpus from ChromaDB
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    init_bm25_index(corpus)
