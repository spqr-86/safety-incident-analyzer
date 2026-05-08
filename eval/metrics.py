"""Deterministic evaluation metrics for V7/V8 RAG pipeline.

No LLM calls. Uses pymorphy3 lemmatization (same as nlp_core.extract_keywords).

Public API:
    extract_key_phrases(text) -> set[str]
    compute_completeness(ground_truth, answer) -> float
    compute_abstain_rate(results) -> float
    compute_false_abstain_rate(results) -> float
    compute_correct_abstain_rate(results) -> float
    compute_retrieval_stats(results) -> dict
"""

from __future__ import annotations

from src.v7.nlp_core import extract_keywords


def extract_key_phrases(text: str) -> set[str]:
    """Ключевые леммы из текста (обёртка над nlp_core.extract_keywords)."""
    return extract_keywords(text)


def compute_completeness(ground_truth: str, answer: str) -> float:
    """Доля ключевых фраз ground_truth, найденных в ответе.

    Алгоритм:
    1. Извлечь ключевые леммы из ground_truth
    2. Извлечь ключевые леммы из answer
    3. completeness = |intersection| / |gt_lemmas|

    Returns:
        float in [0.0, 1.0]. 1.0 если ground_truth пустой (нечего проверять).
    """
    gt_lemmas = extract_key_phrases(ground_truth)
    if not gt_lemmas:
        return 1.0
    if not answer:
        return 0.0
    answer_lemmas = extract_key_phrases(answer)
    found = len(gt_lemmas & answer_lemmas)
    return found / len(gt_lemmas)


def _is_abstained(result: dict) -> bool:
    """True если ответ отсутствует или явно помечен как abstained."""
    if result.get("abstained") is True:
        return True
    return not bool(result.get("answer", "").strip())


def compute_abstain_rate(results: list[dict]) -> float:
    """Доля запросов, на которые система отказала отвечать.

    Args:
        results: список dict с полями "answer" (str) и опционально "abstained" (bool)

    Returns:
        float in [0.0, 1.0]
    """
    if not results:
        return 0.0
    abstained = sum(1 for r in results if _is_abstained(r))
    return abstained / len(results)


def compute_false_abstain_rate(results: list[dict]) -> float:
    """Доля domain-запросов (is_oos=False), на которые система отказала.

    False abstain = система промолчала на вопрос который должна была ответить.
    Цель: = 0.

    Args:
        results: список dict с полями "answer", "is_oos" (bool)
    """
    if not results:
        return 0.0
    domain_results = [r for r in results if not r.get("is_oos", False)]
    if not domain_results:
        return 0.0
    false_abstains = sum(1 for r in domain_results if _is_abstained(r))
    return false_abstains / len(domain_results)


def compute_correct_abstain_rate(results: list[dict]) -> float:
    """Доля OOS-запросов (is_oos=True), на которые система корректно отказала.

    Цель: = 1.0 (система всегда молчит на "как приготовить борщ").

    Args:
        results: список dict с полями "answer", "is_oos" (bool)
    """
    if not results:
        return 1.0
    oos_results = [r for r in results if r.get("is_oos", False)]
    if not oos_results:
        return 1.0  # нет OOS → условие выполнено тривиально
    correct = sum(1 for r in oos_results if _is_abstained(r))
    return correct / len(oos_results)


def compute_retrieval_stats(results: list[dict]) -> dict:
    """Средние показатели retrieval по всем запросам.

    Args:
        results: список dict с полями "top_score" (float), "passage_count" (int)

    Returns:
        dict с ключами "avg_top_score", "avg_passage_count"
    """
    if not results:
        return {"avg_top_score": 0.0, "avg_passage_count": 0.0}

    top_scores = [r.get("top_score", 0.0) for r in results]
    passage_counts = [r.get("passage_count", 0) for r in results]
    n = len(results)

    return {
        "avg_top_score": round(sum(top_scores) / n, 4),
        "avg_passage_count": round(sum(passage_counts) / n, 2),
    }
