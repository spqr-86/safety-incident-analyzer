"""
Demo скрипт для демонстрации работы метрик.

Быстрая демонстрация без необходимости реального RAG inference.
Использует mock данные для показа возможностей evaluation системы.

Использование:
    python scripts/demo_metrics.py
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval_metrics import (
    evaluate_retrieval,
    evaluate_retrieval_batch,
)
from src.advanced_generation_metrics import (
    extract_citations,
    evaluate_citation_quality,
)


def demo_retrieval_metrics():
    """Демонстрация retrieval метрик."""
    print("\n" + "=" * 70)
    print("📊 ДЕМО: Retrieval Metrics")
    print("=" * 70)

    # Пример 1: Успешный retrieval
    print("\n✅ Сценарий 1: Хороший retrieval")
    print("-" * 70)

    retrieved = ["doc_1", "doc_5", "doc_3", "doc_7", "doc_2"]
    relevant = ["doc_1", "doc_2", "doc_3"]

    print(f"Retrieved docs: {retrieved[:5]}")
    print(f"Relevant docs:  {relevant}")

    metrics = evaluate_retrieval(retrieved, relevant, k=5)
    print("\nРезультаты:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Пример 2: Плохой retrieval
    print("\n\n❌ Сценарий 2: Плохой retrieval")
    print("-" * 70)

    retrieved_bad = ["doc_99", "doc_88", "doc_77", "doc_1", "doc_66"]
    relevant_bad = ["doc_1", "doc_2", "doc_3"]

    print(f"Retrieved docs: {retrieved_bad[:5]}")
    print(f"Relevant docs:  {relevant_bad}")

    metrics_bad = evaluate_retrieval(retrieved_bad, relevant_bad, k=5)
    print("\nРезультаты:")
    for metric, value in metrics_bad.items():
        print(f"  {metric}: {value:.3f}")

    # Пример 3: Batch evaluation с MRR
    print("\n\n📊 Сценарий 3: Batch evaluation (MRR)")
    print("-" * 70)

    retrieved_batch = [
        ["doc_1", "doc_2", "doc_3"],  # Релевантный на позиции 1 (RR=1.0)
        ["doc_5", "doc_1", "doc_3"],  # Релевантный на позиции 2 (RR=0.5)
        ["doc_7", "doc_8", "doc_1"],  # Релевантный на позиции 3 (RR=0.33)
    ]
    relevant_batch = [["doc_1"], ["doc_1"], ["doc_1"]]

    batch_metrics = evaluate_retrieval_batch(retrieved_batch, relevant_batch, k=3)

    print("3 запроса:")
    for i, (ret, rel) in enumerate(zip(retrieved_batch, relevant_batch), 1):
        pos = ret.index(rel[0]) + 1 if rel[0] in ret else -1
        rr = 1 / pos if pos > 0 else 0.0
        print(f"  Запрос {i}: {rel[0]} на позиции {pos} (RR={rr:.2f})")

    print("\nСредние метрики:")
    for metric, value in batch_metrics.items():
        print(f"  {metric}: {value:.3f}")


def demo_generation_metrics():
    """Демонстрация generation метрик."""
    print("\n" + "=" * 70)
    print("🎯 ДЕМО: Generation Metrics")
    print("=" * 70)

    # Пример 1: Хорошая цитирование
    print("\n✅ Сценарий 1: Ответ с цитатами")
    print("-" * 70)

    answer_good = """
    По программе А обучаются работодатели (руководители), заместители
    руководителя по охране труда, руководители филиалов [cite: 140, 141].
    Также обязательно обучение специалистов по охране труда [cite: 143, 146].
    """

    citations = extract_citations(answer_good)
    print(f"Текст ответа: {answer_good.strip()[:100]}...")
    print(f"\nИзвлеченные цитаты: {citations}")

    cite_metrics = evaluate_citation_quality(answer_good, "контекст", [])
    print("\nМетрики цитирования:")
    print(f"  Есть цитаты: {cite_metrics['has_citations']}")
    print(f"  Количество цитат: {cite_metrics['citation_count']}")
    print(f"  Уникальных цитат: {cite_metrics['unique_citation_count']}")
    print(f"  Citation diversity: {cite_metrics['citation_diversity']:.2f}")

    # Пример 2: Без цитат
    print("\n\n❌ Сценарий 2: Ответ без цитат")
    print("-" * 70)

    answer_bad = """
    По программе А обучаются работодатели и специалисты.
    """

    citations_bad = extract_citations(answer_bad)
    print(f"Текст ответа: {answer_bad.strip()}")
    print(f"\nИзвлеченные цитаты: {citations_bad}")

    cite_metrics_bad = evaluate_citation_quality(answer_bad, "контекст", [])
    print("\nМетрики цитирования:")
    print(f"  Есть цитаты: {cite_metrics_bad['has_citations']}")
    print(f"  Количество цитат: {cite_metrics_bad['citation_count']}")

    # Пример 3: Повторяющиеся цитаты
    print("\n\n⚠️  Сценарий 3: Повторяющиеся цитаты")
    print("-" * 70)

    answer_dup = """
    Первое утверждение [cite: 100].
    Второе утверждение [cite: 100].
    Третье утверждение [cite: 101].
    """

    citations_dup = extract_citations(answer_dup)
    print(f"Текст ответа: {answer_dup.strip()}")
    print(f"\nВсе цитаты: {citations_dup}")

    cite_metrics_dup = evaluate_citation_quality(answer_dup, "контекст", [])
    print("\nМетрики цитирования:")
    print(f"  Всего цитат: {cite_metrics_dup['citation_count']}")
    print(f"  Уникальных: {cite_metrics_dup['unique_citation_count']}")
    print(f"  Citation diversity: {cite_metrics_dup['citation_diversity']:.2f}")
    print("\n  💡 Diversity < 1.0 означает повторяющиеся цитаты")


def demo_comparison_scenarios():
    """Демонстрация сравнительных сценариев."""
    print("\n" + "=" * 70)
    print("🆚 ДЕМО: Сравнение различных стратегий")
    print("=" * 70)

    print("\n📊 Сравнение 3 retrieval стратегий:")
    print("-" * 70)

    # Общий ground truth
    relevant = ["doc_target_1", "doc_target_2", "doc_target_3"]

    # Стратегия A: Векторный поиск
    strategy_a = [
        "doc_target_1",
        "doc_noise_1",
        "doc_target_2",
        "doc_noise_2",
        "doc_target_3",
    ]

    # Стратегия B: Гибридный (векторный + BM25)
    strategy_b = [
        "doc_target_1",
        "doc_target_2",
        "doc_target_3",
        "doc_noise_1",
        "doc_noise_2",
    ]

    # Стратегия C: Только BM25
    strategy_c = [
        "doc_noise_1",
        "doc_target_1",
        "doc_noise_2",
        "doc_target_2",
        "doc_noise_3",
    ]

    strategies = {
        "Векторный": strategy_a,
        "Гибридный": strategy_b,
        "BM25": strategy_c,
    }

    print(f"\nРелевантные документы: {relevant}\n")

    for name, retrieved in strategies.items():
        metrics = evaluate_retrieval(retrieved, relevant, k=5)
        print(f"{name}:")
        print(f"  Retrieved: {retrieved}")
        print(f"  Hit Rate: {metrics['hit_rate@5']:.2f}")
        print(f"  Precision: {metrics['precision@5']:.2f}")
        print(f"  Recall: {metrics['recall@5']:.2f}")
        print()

    print("💡 Вывод: Гибридный подход показывает лучшие результаты!")


def main():
    """Запуск демонстрации."""
    print("\n" + "=" * 70)
    print("🎭 ДЕМОНСТРАЦИЯ EVALUATION СИСТЕМЫ")
    print("=" * 70)
    print("\nЭто demo показывает работу метрик на mock данных.")
    print("Для реальной оценки используйте: python eval/run_full_evaluation.py")

    # Retrieval метрики
    demo_retrieval_metrics()

    # Generation метрики
    demo_generation_metrics()

    # Сравнительные сценарии
    demo_comparison_scenarios()

    # Финал
    print("\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)
    print("\n📚 Дополнительные ресурсы:")
    print("  • Полная eval: python eval/run_full_evaluation.py")
    print("  • Примеры: docs/evaluation/examples.md")
    print("  • Quick Start: docs/guides/quick-start.md")
    print("  • Unit тесты: python -m pytest tests/test_retrieval_metrics.py -v")
    print("\n")


if __name__ == "__main__":
    main()
