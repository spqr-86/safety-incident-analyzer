"""Pure helper functions for Streamlit UI — no st imports in functions."""

from __future__ import annotations

import os
import re
from typing import List

PROOF_IMAGE_PATTERN = re.compile(r"(static/visuals/proof_[a-f0-9]+\.png)")


def find_proof_images(text: str) -> List[str]:
    """Extract proof image paths from text. Returns only paths that exist on disk."""
    return [p for p in PROOF_IMAGE_PATTERN.findall(text) if os.path.exists(p)]
