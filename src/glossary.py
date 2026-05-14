"""Term glossary: deterministic expansion of domain abbreviations in queries.

Bridges short labels users type ("программа Б") to the official terms the corpus
uses ("программа обучения безопасным методам и приёмам..."), so semantic + BM25
retrieval can match the right chunks. Used by both the V7 pipeline and the legacy
multiagent_rag agent.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

GLOSSARY_PATH = Path(__file__).parent.parent / "config" / "term_glossary.yaml"


@lru_cache(maxsize=2)
def load_glossary(path: str = str(GLOSSARY_PATH)) -> dict[str, str]:
    """Load {short_term: official_term} from the glossary YAML. Cached per path."""
    p = Path(path)
    if not p.exists():
        logger.warning("Term glossary not found at %s", p)
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        terms = data.get("terms", {}) if data else {}
        return {
            key.lower(): val["official"]
            for key, val in terms.items()
            if "official" in val
        }
    except Exception as e:
        logger.error("Failed to load glossary: %s", e)
        return {}


def _make_term_pattern(term: str) -> re.Pattern:
    """Build a regex matching the term as whole words, tolerating Russian
    inflectional endings on words longer than 4 chars.

    Words >4 chars: drop the last 2 chars, allow a bounded inflectional suffix.
    Words <=4 chars: exact word match. Stemming short abbreviations is
    catastrophic — "соут" -> r"со\\w*" matches "состоять", "создание", etc.
    Every part is anchored with \\b so it never matches mid-word.
    Example: "программа а" -> r"\\bпрограм\\w{0,3}\\b\\s+\\bа\\b".
    """
    parts = []
    for w in term.lower().split():
        if len(w) > 4:
            parts.append(r"\b" + re.escape(w[: len(w) - 2]) + r"\w{0,3}\b")
        else:
            parts.append(r"\b" + re.escape(w) + r"\b")
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


@lru_cache(maxsize=1)
def _compiled_patterns() -> list[tuple[re.Pattern, str, str]]:
    """Pre-compile (pattern, short_term, official) for every glossary entry."""
    return [
        (_make_term_pattern(short), short, official)
        for short, official in load_glossary().items()
    ]


def expand_query_with_glossary(query: str) -> str:
    """Append official terms for any glossary abbreviations found in the query.

    Returns the query unchanged when nothing matches.
    """
    patterns = _compiled_patterns()
    if not patterns:
        return query
    expansions = [
        f"{short} → {official}"
        for pattern, short, official in patterns
        if pattern.search(query)
    ]
    if expansions:
        return query + "\n\n[Глоссарий: " + "; ".join(expansions) + "]"
    return query
