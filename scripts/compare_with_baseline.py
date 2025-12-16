"""
Скрипт для сравнения текущих метрик с baseline.

Использование:
    python scripts/compare_with_baseline.py benchmarks/results_history.jsonl
    python scripts/compare_with_baseline.py --baseline benchmarks/baseline.json --current benchmarks/results_history.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime


def load_baseline(baseline_path: str) -> Dict[str, Any]:
    """Загружает baseline метрики из JSON файла."""
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_result(results_path: str) -> Dict[str, Any]:
    """Загружает последний результат из JSONL файла."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise ValueError(f"Нет результатов в {results_path}")

    # Возвращаем самый последний результат
    return results[-1]


def compare_metrics(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сравнивает текущие метрики с baseline.

    Returns:
        Dict с результатами сравнения
    """
    comparison = {
        "timestamp": current.get("timestamp", datetime.now().isoformat()),
        "baseline_date": baseline.get("date", "unknown"),
        "improvements": [],
        "regressions": [],
        "stable": [],
        "summary": {},
    }

    # Метрики для сравнения
    metrics_to_compare = [
        ("mean_correctness_score", "Correctness", 10.0, "higher_better"),
        ("mean_faithfulness_score", "Faithfulness", 1.0, "higher_better"),
        ("mean_answer_relevance_score", "Answer Relevance", 1.0, "higher_better"),
        ("citation_rate", "Citation Rate", 1.0, "higher_better"),
        ("p95_total_time", "P95 Latency", None, "lower_better"),
    ]

    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("aggregate_metrics", {})

    for metric_key, metric_name, max_value, direction in metrics_to_compare:
        baseline_value = baseline_metrics.get(metric_key, 0.0)
        current_value = current_metrics.get(metric_key, 0.0)

        # Вычисляем изменение
        if baseline_value > 0:
            change_pct = ((current_value - baseline_value) / baseline_value) * 100
        else:
            change_pct = 0.0

        # Определяем улучшение или регрессию
        if direction == "higher_better":
            is_improvement = current_value > baseline_value
            is_regression = current_value < baseline_value * 0.95  # 5% порог
        else:  # lower_better
            is_improvement = current_value < baseline_value
            is_regression = current_value > baseline_value * 1.05  # 5% порог

        result = {
            "metric": metric_name,
            "baseline": baseline_value,
            "current": current_value,
            "change": current_value - baseline_value,
            "change_pct": change_pct,
            "direction": direction,
        }

        if is_improvement and abs(change_pct) > 1.0:  # минимум 1% изменение
            comparison["improvements"].append(result)
        elif is_regression:
            comparison["regressions"].append(result)
        else:
            comparison["stable"].append(result)

    # Summary
    comparison["summary"] = {
        "total_metrics": len(metrics_to_compare),
        "improvements": len(comparison["improvements"]),
        "regressions": len(comparison["regressions"]),
        "stable": len(comparison["stable"]),
        "overall_status": "✅ IMPROVED" if len(comparison["improvements"]) > len(comparison["regressions"]) else (
            "❌ REGRESSED" if len(comparison["regressions"]) > 0 else "➖ STABLE"
        ),
    }

    return comparison


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Выводит результаты сравнения в консоль."""
    print("\n" + "=" * 70)
    print("📊 СРАВНЕНИЕ С BASELINE")
    print("=" * 70)

    print(f"\n📅 Baseline дата: {comparison['baseline_date']}")
    print(f"📅 Текущий запуск: {comparison['timestamp']}")

    print(f"\n🎯 Общий статус: {comparison['summary']['overall_status']}")
    print(f"   Улучшений: {comparison['summary']['improvements']}")
    print(f"   Регрессий: {comparison['summary']['regressions']}")
    print(f"   Стабильно: {comparison['summary']['stable']}")

    # Улучшения
    if comparison["improvements"]:
        print("\n✅ УЛУЧШЕНИЯ:")
        for item in comparison["improvements"]:
            sign = "+" if item["change"] > 0 else ""
            print(f"   • {item['metric']}")
            print(f"     Baseline: {item['baseline']:.3f}")
            print(f"     Текущее:  {item['current']:.3f}")
            print(f"     Изменение: {sign}{item['change']:.3f} ({sign}{item['change_pct']:.1f}%)")

    # Регрессии
    if comparison["regressions"]:
        print("\n❌ РЕГРЕССИИ:")
        for item in comparison["regressions"]:
            sign = "+" if item["change"] > 0 else ""
            print(f"   • {item['metric']}")
            print(f"     Baseline: {item['baseline']:.3f}")
            print(f"     Текущее:  {item['current']:.3f}")
            print(f"     Изменение: {sign}{item['change']:.3f} ({sign}{item['change_pct']:.1f}%)")

    # Стабильные
    if comparison["stable"]:
        print("\n➖ СТАБИЛЬНЫЕ (изменение < 1%):")
        for item in comparison["stable"]:
            print(f"   • {item['metric']}: {item['current']:.3f}")

    print("\n" + "=" * 70)


def save_comparison(comparison: Dict[str, Any], output_path: str) -> None:
    """Сохраняет результаты сравнения в JSON файл."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Результаты сравнения сохранены в: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Сравнить текущие метрики с baseline"
    )
    parser.add_argument(
        "results",
        nargs="?",
        default="benchmarks/results_history.jsonl",
        help="Путь к файлу с результатами (JSONL)",
    )
    parser.add_argument(
        "--baseline",
        default="benchmarks/baseline.json",
        help="Путь к baseline файлу (JSON)",
    )
    parser.add_argument(
        "--output",
        help="Путь для сохранения результатов сравнения",
    )

    args = parser.parse_args()

    # Проверяем существование файлов
    baseline_path = Path(args.baseline)
    results_path = Path(args.results)

    if not baseline_path.exists():
        print(f"❌ Baseline файл не найден: {baseline_path}")
        print("💡 Создайте baseline файл, запустив: python eval/run_full_evaluation.py")
        sys.exit(1)

    if not results_path.exists():
        print(f"❌ Файл с результатами не найден: {results_path}")
        print("💡 Запустите eval: python eval/run_full_evaluation.py")
        sys.exit(1)

    # Загружаем данные
    print("📂 Загрузка данных...")
    baseline = load_baseline(str(baseline_path))
    current = load_latest_result(str(results_path))

    # Сравниваем
    comparison = compare_metrics(baseline, current)

    # Выводим результаты
    print_comparison(comparison)

    # Сохраняем если указан output
    if args.output:
        save_comparison(comparison, args.output)

    # Exit code на основе результата
    if comparison["summary"]["regressions"] > 0:
        print("\n⚠️  Обнаружены регрессии метрик!")
        sys.exit(1)
    else:
        print("\n✅ Метрики в норме или улучшились!")
        sys.exit(0)


if __name__ == "__main__":
    main()
