# src/v7/nlp_core.py
# V7 Migration — Stage 1: NLP Core

from typing import List
import pymorphy3
import razdel
from rank_bm25 import BM25Okapi
import numpy as np
import re

from src.v7.state_types import Doc, ScoredDoc


# Placeholder for stop words, will be replaced with a more robust list
RUSSIAN_STOP_WORDS = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
]

morph = pymorphy3.MorphAnalyzer()

import re

# ... (imports) ...

def extract_keywords(text: str) -> List[str]:
    """Extracts and lemmatizes keywords from Russian text."""
    # Remove punctuation and convert to lower case
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = [token.text for token in razdel.tokenize(text)]
    lemmatized_tokens = []
    for token in tokens:
        parsed_token = morph.parse(token)[0]
        if parsed_token.normal_form not in RUSSIAN_STOP_WORDS and parsed_token.tag.POS not in {"PREP", "CONJ", "PRCL", "INTJ"}:
            lemmatized_tokens.append(parsed_token.normal_form)
    return list(set(lemmatized_tokens))


class BM25Index:
    """A wrapper for rank_bm25.BM25Okapi."""
    def __init__(self):
        self.bm25 = None
        self.docs = []

    def build(self, docs: List[Doc]):
        """Builds the BM25 index from a list of documents."""
        self.docs = docs
        tokenized_corpus = [extract_keywords(doc.text) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query(self, tokens: List[str], top_k: int) -> List[ScoredDoc]:
        """Queries the index and returns top_k scored documents."""
        if self.bm25 is None:
            return []
        doc_scores = self.bm25.get_scores(tokens)
        # Set scores of docs with no matching terms to a very small number
        doc_scores = np.where(doc_scores == 0, -1e9, doc_scores)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            score = doc_scores[i]
            original_doc = self.docs[i]
            results.append(ScoredDoc(id=original_doc.id, text=original_doc.text, metadata=original_doc.metadata, score=score))
        
        return results


def rrf_merge(rankings: List[List[ScoredDoc]], k: int = 60) -> List[ScoredDoc]:
    """Merges multiple ranked lists of documents using Reciprocal Rank Fusion."""
    if not rankings:
        return []

    rrf_scores = {}
    doc_map = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            doc_map[doc.id] = doc
            if doc.id not in rrf_scores:
                rrf_scores[doc.id] = 0.0
            rrf_scores[doc.id] += 1.0 / (k + rank)
    
    # Sort documents by their RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    merged_results = []
    for doc_id in sorted_doc_ids:
        doc = doc_map[doc_id]
        # Create a new ScoredDoc with the RRF score
        merged_results.append(ScoredDoc(id=doc.id, text=doc.text, metadata=doc.metadata, score=rrf_scores[doc_id]))
        
    return merged_results

def mmr_select(docs: List[Doc], query_emb: np.ndarray, lambda_: float, k: int) -> List[Doc]:
    """
    Selects a diverse subset of documents using Maximal Marginal Relevance (MMR).
    Note: This is a placeholder implementation. A real implementation would require
    document embeddings to be pre-calculated and passed in.
    """
    # This is a complex function to implement without pre-existing embeddings.
    # For the purpose of this stage, we will return the first k documents.
    # A proper implementation will be added in a later stage when embeddings are handled.
    if not docs:
        return []
        
    if len(docs) <= k:
        return docs

    return docs[:k]

