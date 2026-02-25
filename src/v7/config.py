# src/v7/config.py
"""V7 RAG pipeline — configuration.

All thresholds and limits externalized via pydantic-settings.
Env vars use V7_ prefix (e.g. V7_HARD_GATE_THRESHOLD=0.80).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class V7Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="V7_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Simple path (rag_simple) ───────────────────────────────────────────
    # ChromaDB L2→similarity formula: 1/(1+d). Relevant docs score ~0.44–0.56,
    # so threshold must be ≤ 0.44. 0.40 passes good matches, rejects garbage (<0.35).
    HARD_GATE_THRESHOLD: float = 0.40  # plan.threshold (similarity hard gate)
    TRIAGE_SOFT_THRESHOLD: float = 0.40  # plan.borderline_threshold
    MIN_PASSAGES: int = 5  # plan.min_passages
    # Russian adj/noun lemmas differ ("лестничный"≠"лестница") → keep low
    MIN_KEYWORD_OVERLAP_ACTIVE: float = 0.15  # plan.min_keyword_overlap
    MAX_SINGLE_DOC_RATIO: float = 0.8  # plan.max_single_doc_ratio
    SIMPLE_TOP_K: int = 12  # plan.top_k
    SIMPLE_TIMEOUT_MS: int = 250  # plan.timeout_ms

    # ── Complex path (rag_complex) ────────────────────────────────────────
    COMPLEX_THRESHOLD: float = 0.35  # min threshold for complex (floor, ≤ simple)
    COMPLEX_MIN_PASSAGES: int = 8  # plan.min_passages for slow path
    # Russian morphology: adjective and noun lemmas often differ for the same root.
    # 0.20 is enough to detect real OOS (0-overlap) without blocking domain queries.
    COMPLEX_MIN_KW_OVERLAP: float = 0.20  # plan.min_keyword_overlap for slow path
    COMPLEX_MAX_SINGLE_DOC_RATIO: float = 0.7
    COMPLEX_BORDERLINE_THRESHOLD: float = 0.30
    COMPLEX_TOP_K: int = 60  # plan.top_k for slow path
    COMPLEX_TIMEOUT_MS: int = 1200

    # ── Retrieval engine ──────────────────────────────────────────────────
    RRF_K: int = 60
    MMR_LAMBDA: float = 0.7
    BM25_TOP_K: int = 20
    SEMANTIC_TOP_K: int = 20

    # ── Keyword overlap (dual) ────────────────────────────────────────────
    MIN_KEYWORD_OVERLAP_ORIGINAL: float = 0.10  # drift detection, even looser

    # ── LLM & Limits ──────────────────────────────────────────────────────
    MAX_REWRITE_ATTEMPTS: int = 2
    MAX_CHUNKS_FOR_LLM: int = 10
    VERIFIER_CONFIDENCE_ANCHOR: float = 0.7  # plan.min_verifier_confidence

    # ── Anti-injection ────────────────────────────────────────────────────
    COVERAGE_DROP_PCT: float = 0.30
    MAX_INPUT_LENGTH: int = 2000
    BLOCKED_PATTERNS: list[str] = [
        "ignore previous",
        "system prompt",
        "you are now",
    ]


v7_config = V7Config()
