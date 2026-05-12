"""Deterministic evaluation metrics for V7/V8 RAG pipeline.

No LLM calls. Uses pymorphy3 lemmatization (same as nlp_core.extract_keywords).

Public API:
    extract_key_phrases(text) -> set[str]
    compute_completeness(ground_truth, answer) -> float
    compute_abstain_rate(results) -> float
    compute_false_abstain_rate(results) -> float
    compute_correct_abstain_rate(results) -> float
    compute_retrieval_stats(results) -> dict
    compute_inversion_detected(must_not_contain, answer) -> bool
    compute_inversion_rate(results) -> float
    parse_citations(answer) -> list[dict]
    compute_citation_rate(answer) -> float
    compute_citation_in_retrieval(answer, passages) -> float
    compute_citation_doc_match(answer, passages) -> float
"""

from __future__ import annotations

import re

from src.v7.nlp_core import extract_keywords

# Regex for [Фрагмент N: Документ, п. X.X] or [Фрагмент N: Документ, без пункта]
_CITATION_RE = re.compile(
    r"\[Фрагмент\s+(\d+):\s*([^,\]]+?)(?:,\s*п\.\s*([^\]]+?)|,\s*без\s+пункта)?\s*\]",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


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


def compute_inversion_detected(must_not_contain: str, answer: str) -> bool:
    """True если ответ содержит запрещённый паттерн (инверсия нормы).

    Инверсия = система указала неверное числовое значение или норму.
    Пример: "повторный инструктаж раз в год" вместо "раз в 6 месяцев".

    Args:
        must_not_contain: паттерны через '|' (case-insensitive substring match).
                          Пустая строка → проверка не выполняется.
        answer: ответ системы.

    Returns:
        True = инверсия обнаружена (плохо). False = всё чисто или проверка не нужна.
    """
    if not must_not_contain or not must_not_contain.strip() or not answer:
        return False
    answer_lower = answer.lower()
    patterns = [p.strip() for p in must_not_contain.split("|") if p.strip()]
    return any(p.lower() in answer_lower for p in patterns)


def compute_inversion_rate(results: list[dict]) -> float:
    """Доля ответов с инверсиями среди строк с заполненным must_not_contain.

    Строки без must_not_contain не учитываются.

    Args:
        results: список dict с полями "must_not_contain" (str) и "inversion_detected" (bool).

    Returns:
        float in [0.0, 1.0]. 0.0 если нет проверяемых строк.
    """
    checkable = [r for r in results if r.get("must_not_contain", "").strip()]
    if not checkable:
        return 0.0
    inversions = sum(1 for r in checkable if r.get("inversion_detected", False))
    return inversions / len(checkable)


def parse_citations(answer: str) -> list[dict]:
    """Извлечь ссылки формата [Фрагмент N: Документ, п. X.X] из ответа.

    Returns:
        Список dict с ключами:
            n (int)       — номер фрагмента
            doc (str)     — название документа
            section (str) — номер пункта или "" если "без пункта"
    """
    if not answer:
        return []
    results = []
    for m in _CITATION_RE.finditer(answer):
        results.append(
            {
                "n": int(m.group(1)),
                "doc": m.group(2).strip(),
                "section": (m.group(3) or "").strip(),
            }
        )
    return results


def _split_sentences(text: str) -> list[str]:
    """Разбить текст на предложения, не разрезая содержимое [...].

    Маскируем цитаты перед split чтобы точки внутри [Фрагмент N: Doc, п. X.X]
    не ломали границы предложений.
    """
    # Заменяем каждый символ внутри [...] на 'X' (длина сохраняется)
    masked = re.sub(r"\[[^\]]*\]", lambda m: "X" * len(m.group()), text)
    parts: list[str] = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(masked):
        end = m.end()
        parts.append(text[last:end].strip())
        last = end
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            parts.append(tail)
    return [p for p in parts if p]


def compute_citation_rate(answer: str) -> float:
    """Доля предложений с хотя бы одной ссылкой.

    Предложения определяются по знакам .!? вне квадратных скобок.

    Returns:
        float in [0.0, 1.0]. 0.0 если нет предложений или ответ пустой.
    """
    if not answer:
        return 0.0
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    with_citation = sum(1 for s in sentences if _CITATION_RE.search(s))
    return with_citation / len(sentences)


def compute_citation_in_retrieval(answer: str, passages: list[dict]) -> float:
    """Доля ссылок где N находится в диапазоне [1, len(passages)].

    Защита от выдуманных номеров фрагментов (N > кол-ва реальных пассажей).

    Args:
        answer: ответ системы.
        passages: список пассажей, возвращённых из final_passages.

    Returns:
        float in [0.0, 1.0]. 0.0 если нет ссылок или passages пустой.
    """
    citations = parse_citations(answer)
    if not citations or not passages:
        return 0.0
    n_passages = len(passages)
    valid = sum(1 for c in citations if 1 <= c["n"] <= n_passages)
    return valid / len(citations)


def compute_citation_doc_match(answer: str, passages: list[dict]) -> float:
    """Доля ссылок где название документа совпадает с источником фрагмента N.

    Matching: cited_doc является подстрокой source (case-insensitive).
    Пропускает ссылки с N вне диапазона passages.

    Args:
        answer: ответ системы.
        passages: список пассажей с metadata.source.

    Returns:
        float in [0.0, 1.0]. 0.0 если нет проверяемых ссылок.
    """
    citations = parse_citations(answer)
    if not citations or not passages:
        return 0.0
    checkable = [c for c in citations if 1 <= c["n"] <= len(passages)]
    if not checkable:
        return 0.0
    matched = 0
    for c in checkable:
        source = passages[c["n"] - 1].get("metadata", {}).get("source", "").lower()
        cited_doc = c["doc"].lower().strip()
        if cited_doc and cited_doc in source:
            matched += 1
    return matched / len(checkable)


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
