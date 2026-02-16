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

    # Hard gates
    HARD_GATE_THRESHOLD: float = 0.65
    TRIAGE_SOFT_THRESHOLD: float = 0.40
    COVERAGE_DROP_PCT: float = 0.30

    # Retrieval
    RRF_K: int = 60
    MMR_LAMBDA: float = 0.7
    BM25_TOP_K: int = 20
    SEMANTIC_TOP_K: int = 20

    # Keyword overlap (dual)
    MIN_KEYWORD_OVERLAP_ACTIVE: float = 0.3
    MIN_KEYWORD_OVERLAP_ORIGINAL: float = 0.2

    # LLM & Limits
    MAX_REWRITE_ATTEMPTS: int = 2
    MAX_CHUNKS_FOR_LLM: int = 10
    VERIFIER_CONFIDENCE_ANCHOR: float = 0.7

    # Anti-injection
    MAX_INPUT_LENGTH: int = 2000
    BLOCKED_PATTERNS: list[str] = [
        "ignore previous",
        "system prompt",
        "you are now",
    ]


v7_config = V7Config()
