"""
Анализ трендов метрик из истории запусков eval.

Использование:
    python scripts/analyze_trends.py
    python scripts/analyze_trends.py --limit 10  # Последние 10 запусков
    python scripts/analyze_trends.py --export benchmarks/trends.csv
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import csv


def load_all_results(results_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Загружает все результаты из JSONL файла."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if limit:
        results = results[-limit:]

    return results


def calculate_trend(values: List[float]) -> Dict[str, Any]:
    """
    Вычисляет тренд для списка значений.

    Returns:
        Dict с информацией о тренде
    """
    if len(values) < 2:
        return {
            "direction": "insufficient_data",
            "change": 0.0,
            "change_pct": 0.0,
            "emoji": "➖",
        }

    first_value = values[0]
    last_value = values[-1]
    change = last_value - first_value

    if first_value != 0:
        change_pct = (change / first_value) * 100
    else:
        change_pct = 0.0

    # Определяем направление
    if abs(change_pct) < 1.0:
        direction = "stable"
        emoji = "➖"
    elif change_pct > 0:
        direction = "improving"
        emoji = "📈"
    else:
        direction = "declining"
        emoji = "📉"

    return {
        "direction": direction,
        "change": change,
        "change_pct": change_pct,
        "emoji": emoji,
        "first": first_value,
        "last": last_value,
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def analyze_metrics_trends(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Анализирует тренды для всех метрик."""
    # Извлекаем временные ряды для каждой метрики
    metrics_timeseries = {}

    for result in results:
        timestamp = result.get("timestamp", "")
        metrics = result.get("aggregate_metrics", {})

        for metric_key, value in metrics.items():
            if metric_key not in metrics_timeseries:
                metrics_timeseries[metric_key] = {
                    "timestamps": [],
                    "values": [],
                }

            metrics_timeseries[metric_key]["timestamps"].append(timestamp)
            metrics_timeseries[metric_key]["values"].append(value)

    # Вычисляем тренды
    trends = {}
    for metric_key, timeseries in metrics_timeseries.items():
        trends[metric_key] = calculate_trend(timeseries["values"])
        trends[metric_key]["timestamps"] = timeseries["timestamps"]
        trends[metric_key]["values"] = timeseries["values"]

    return trends


def print_trends_report(
    trends: Dict[str, Any],
    results_count: int,
    metric_names: Dict[str, str] = None,
) -> None:
    """Выводит отчет по трендам."""
    if metric_names is None:
        metric_names = {
            "mean_correctness_score": "Correctness",
            "mean_faithfulness_score": "Faithfulness",
            "mean_answer_relevance_score": "Answer Relevance",
            "citation_rate": "Citation Rate",
            "p95_total_time": "P95 Latency",
            "mean_total_time": "Avg Latency",
        }

    print("\n" + "=" * 70)
    print("📈 АНАЛИЗ ТРЕНДОВ МЕТРИК")
    print("=" * 70)

    print(f"\n📊 Анализ на основе: {results_count} запусков eval")

    # Группируем по направлению тренда
    improving = []
    declining = []
    stable = []

    for metric_key, trend_data in trends.items():
        metric_name = metric_names.get(metric_key, metric_key)

        if trend_data["direction"] == "improving":
            improving.append((metric_name, trend_data))
        elif trend_data["direction"] == "declining":
            declining.append((metric_name, trend_data))
        elif trend_data["direction"] == "stable":
            stable.append((metric_name, trend_data))

    # Улучшающиеся метрики
    if improving:
        print(f"\n📈 УЛУЧШАЮЩИЕСЯ МЕТРИКИ ({len(improving)}):")
        for metric_name, trend in improving:
            print(f"\n   {trend['emoji']} {metric_name}")
            print(f"      Первое значение:  {trend['first']:.3f}")
            print(f"      Последнее:        {trend['last']:.3f}")
            print(
                f"      Изменение:        +{trend['change']:.3f} (+{trend['change_pct']:.1f}%)"
            )
            print(
                f"      Мин/Макс/Средн:   {trend['min']:.3f} / {trend['max']:.3f} / {trend['avg']:.3f}"
            )

    # Ухудшающиеся метрики
    if declining:
        print(f"\n📉 УХУДШАЮЩИЕСЯ МЕТРИКИ ({len(declining)}):")
        for metric_name, trend in declining:
            print(f"\n   {trend['emoji']} {metric_name}")
            print(f"      Первое значение:  {trend['first']:.3f}")
            print(f"      Последнее:        {trend['last']:.3f}")
            print(
                f"      Изменение:        {trend['change']:.3f} ({trend['change_pct']:.1f}%)"
            )
            print(
                f"      Мин/Макс/Средн:   {trend['min']:.3f} / {trend['max']:.3f} / {trend['avg']:.3f}"
            )

    # Стабильные метрики
    if stable:
        print(f"\n➖ СТАБИЛЬНЫЕ МЕТРИКИ ({len(stable)}):")
        for metric_name, trend in stable:
            print(
                f"   • {metric_name}: {trend['avg']:.3f} (вариация: {abs(trend['change_pct']):.1f}%)"
            )

    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    if declining:
        print("   ⚠️  Обратите внимание на ухудшающиеся метрики")
        print("   • Проанализируйте последние изменения кода")
        print("   • Проверьте не изменились ли параметры конфигурации")
        print("   • Рассмотрите откат к предыдущей версии")
    elif improving:
        print("   ✅ Метрики улучшаются!")
        print("   • Зафиксируйте текущие параметры как baseline")
        print("   • Продолжайте мониторинг для подтверждения тренда")
    else:
        print("   ➖ Метрики стабильны")
        print("   • Рассмотрите эксперименты с новыми подходами")
        print("   • Проведите hyperparameter optimization")

    print("\n" + "=" * 70)


def export_to_csv(trends: Dict[str, Any], output_path: str) -> None:
    """Экспортирует тренды в CSV."""
    # Находим максимальное количество точек
    max_points = max(len(t["values"]) for t in trends.values())

    # Подготавливаем данные
    rows = []
    for i in range(max_points):
        row = {}
        for metric_key, trend_data in trends.items():
            if i < len(trend_data["values"]):
                row[f"{metric_key}"] = trend_data["values"][i]
                if i < len(trend_data["timestamps"]):
                    row["timestamp"] = trend_data["timestamps"][i]
        rows.append(row)

    # Записываем CSV
    if rows:
        fieldnames = ["timestamp"] + list(trends.keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n💾 Данные экспортированы в CSV: {output_path}")
        print(f"   Строк: {len(rows)}")
        print(f"   Метрик: {len(trends)}")


def main():
    parser = argparse.ArgumentParser(description="Анализ трендов метрик")
    parser.add_argument(
        "--results",
        default="benchmarks/results_history.jsonl",
        help="Путь к файлу с результатами",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Ограничить количество последних запусков для анализа",
    )
    parser.add_argument(
        "--export",
        help="Экспортировать в CSV",
    )

    args = parser.parse_args()

    # Проверяем существование файла
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Файл с результатами не найден: {results_path}")
        print("💡 Запустите сначала: python eval/run_full_evaluation.py")
        sys.exit(1)

    # Загружаем результаты
    print("📂 Загрузка результатов...")
    results = load_all_results(str(results_path), limit=args.limit)

    if len(results) < 2:
        print("⚠️  Недостаточно данных для анализа трендов (нужно минимум 2 запуска)")
        print(f"   Найдено: {len(results)} запусков")
        sys.exit(1)

    print(f"✅ Загружено {len(results)} запусков")

    # Анализируем тренды
    print("🔍 Анализ трендов...")
    trends = analyze_metrics_trends(results)

    # Выводим отчет
    print_trends_report(trends, len(results))

    # Экспорт в CSV
    if args.export:
        export_to_csv(trends, args.export)


if __name__ == "__main__":
    main()
