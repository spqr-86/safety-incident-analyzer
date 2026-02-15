from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        threshold: float = 0.93,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_file: str = "semantic_cache.json",
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.cache_file = cache_file

        # Initialize storage
        self.sentences: List[str] = []
        self.embeddings: List[List[float]] = []  # Store as lists for JSON serialization
        self.answers: List[str] = []

        # Load model
        # Note: In tests this is mocked. In prod, this loads the model.
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning("Failed to load SentenceTransformer model: %s", e)
            self.model = None

        # Load cache from disk
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sentences = data.get("sentences", [])
                    self.embeddings = data.get("embeddings", [])
                    self.answers = data.get("answers", [])
            except Exception as e:
                logger.error("Failed to load cache: %s", e)

    def save(self):
        """Save cache to disk."""
        data = {
            "sentences": self.sentences,
            "embeddings": self.embeddings,
            "answers": self.answers,
        }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save cache: %s", e)

    def get(self, query: str) -> Optional[str]:
        if not self.sentences or not self.model:
            return None

        try:
            # Encode query
            query_embedding = self.model.encode(query)

            # Compute cosine similarities
            # Convert list of lists to numpy array for efficient computation
            cache_embeddings = np.array(self.embeddings)

            if len(cache_embeddings) == 0:
                return None

            norm_query = np.linalg.norm(query_embedding)
            if norm_query == 0:
                return None

            norm_cache = np.linalg.norm(cache_embeddings, axis=1)
            # Avoid division by zero
            norm_cache[norm_cache == 0] = 1e-9

            dot_products = np.dot(cache_embeddings, query_embedding)
            similarities = dot_products / (norm_cache * norm_query)

            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            if best_score >= self.threshold:
                return self.answers[best_idx]

        except Exception as e:
            logger.error("Error in semantic cache get: %s", e)
            return None

        return None

    def add(self, query: str, answer: str) -> None:
        if not self.model:
            return

        # Check if already exists (exact match)
        if query in self.sentences:
            return

        try:
            embedding = self.model.encode(query)

            # Ensure embedding is a list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            self.sentences.append(query)
            self.embeddings.append(embedding)
            self.answers.append(answer)

            self.save()
        except Exception as e:
            logger.error("Error adding to semantic cache: %s", e)
