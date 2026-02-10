"""
Скрипт для проверки соответствия метрик целевым значениям.

Использование:
    python scripts/check_target_metrics.py
    python scripts/check_target_metrics.py --results benchmarks/results_history.jsonl
    python scripts/check_target_metrics.py --strict  # Exit code 1 если хотя бы одна метрика не достигла цели
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Целевые значения метрик
TARGET_METRICS = {
    "mean_correctness_score": {
        "target": 8.0,
        "critical": 6.0,
        "direction": "higher",
        "unit": "/10",
        "description": "Correctness score",
    },
    "mean_faithfulness_score": {
        "target": 0.90,
        "critical": 0.70,
        "direction": "higher",
        "unit": "",
        "description": "Faithfulness (no hallucinations)",
    },
    "mean_answer_relevance_score": {
        "target": 0.85,
        "critical": 0.70,
        "direction": "higher",
        "unit": "",
        "description": "Answer relevance",
    },
    "citation_rate": {
        "target": 0.95,
        "critical": 0.80,
        "direction": "higher",
        "unit": "",
        "description": "Citation rate",
    },
    "p95_total_time": {
        "target": 10.0,
        "critical": 20.0,
        "direction": "lower",
        "unit": "s",
        "description": "P95 latency",
    },
}


def load_latest_result(results_path: str) -> Dict[str, Any]:
    """Загружает последний результат из JSONL файла."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise ValueError(f"Нет результатов в {results_path}")

    return results[-1]


def check_metric(
    metric_key: str,
    current_value: float,
    config: Dict[str, Any],
) -> Tuple[str, str, str]:
    """
    Проверяет одну метрику.

    Returns:
        Tuple[status, emoji, message]
        status: 'achieved' | 'acceptable' | 'critical'
    """
    target = config["target"]
    critical = config["critical"]
    direction = config["direction"]

    if direction == "higher":
        # Хотим больше значение
        if current_value >= target:
            return "achieved", "✅", f"Достигнуто (>= {target})"
        elif current_value >= critical:
            return "acceptable", "⚠️", f"Приемлемо (>= {critical}, цель: {target})"
        else:
            return "critical", "❌", f"Критично (< {critical})"
    else:  # lower
        # Хотим меньше значение
        if current_value <= target:
            return "achieved", "✅", f"Достигнуто (<= {target})"
        elif current_value <= critical:
            return "acceptable", "⚠️", f"Приемлемо (<= {critical}, цель: {target})"
        else:
            return "critical", "❌", f"Критично (> {critical})"


def check_all_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Проверяет все целевые метрики.

    Returns:
        Dict с результатами проверки
    """
    metrics = results.get("aggregate_metrics", {})
    checks = []

    achieved_count = 0
    acceptable_count = 0
    critical_count = 0

    for metric_key, config in TARGET_METRICS.items():
        current_value = metrics.get(metric_key, 0.0)
        status, emoji, message = check_metric(metric_key, current_value, config)

        checks.append(
            {
                "metric": config["description"],
                "key": metric_key,
                "current": current_value,
                "target": config["target"],
                "critical": config["critical"],
                "unit": config["unit"],
                "status": status,
                "emoji": emoji,
                "message": message,
            }
        )

        if status == "achieved":
            achieved_count += 1
        elif status == "acceptable":
            acceptable_count += 1
        else:
            critical_count += 1

    return {
        "timestamp": results.get("timestamp", "unknown"),
        "dataset_size": results.get("dataset_size", 0),
        "checks": checks,
        "summary": {
            "total": len(TARGET_METRICS),
            "achieved": achieved_count,
            "acceptable": acceptable_count,
            "critical": critical_count,
        },
    }


def print_results(check_results: Dict[str, Any], verbose: bool = True) -> None:
    """Выводит результаты проверки."""
    print("\n" + "=" * 70)
    print("🎯 ПРОВЕРКА ЦЕЛЕВЫХ МЕТРИК")
    print("=" * 70)

    print(f"\n📅 Timestamp: {check_results['timestamp']}")
    print(f"📊 Dataset size: {check_results['dataset_size']}")

    summary = check_results["summary"]
    print("\n📈 Сводка:")
    print(f"   ✅ Достигнуто целевых:  {summary['achieved']}/{summary['total']}")
    print(f"   ⚠️  Приемлемо:          {summary['acceptable']}/{summary['total']}")
    print(f"   ❌ Критичных:          {summary['critical']}/{summary['total']}")

    # Общий статус
    if summary["critical"] > 0:
        overall_status = "❌ КРИТИЧНО"
        overall_color = "\033[91m"  # Red
    elif summary["achieved"] == summary["total"]:
        overall_status = "✅ ВСЕ ЦЕЛИ ДОСТИГНУТЫ"
        overall_color = "\033[92m"  # Green
    elif summary["acceptable"] + summary["achieved"] == summary["total"]:
        overall_status = "⚠️ ПРИЕМЛЕМО"
        overall_color = "\033[93m"  # Yellow
    else:
        overall_status = "⚠️ ТРЕБУЕТСЯ ВНИМАНИЕ"
        overall_color = "\033[93m"  # Yellow

    print(f"\n🎯 Общий статус: {overall_color}{overall_status}\033[0m")

    # Детали по каждой метрике
    if verbose:
        print("\n📊 Детали по метрикам:")
        print("")

        for check in check_results["checks"]:
            print(f"{check['emoji']} {check['metric']}")
            print(f"   Текущее:  {check['current']:.3f}{check['unit']}")
            print(f"   Цель:     {check['target']:.3f}{check['unit']}")
            print(f"   Статус:   {check['message']}")
            print("")

    print("=" * 70)


def save_report(check_results: Dict[str, Any], output_path: str) -> None:
    """Сохраняет отчет в JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(check_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Отчет сохранен: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Проверить соответствие метрик целевым значениям"
    )
    parser.add_argument(
        "--results",
        default="benchmarks/results_history.jsonl",
        help="Путь к файлу с результатами",
    )
    parser.add_argument(
        "--output",
        help="Путь для сохранения отчета (JSON)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Строгий режим: exit code 1 если не все цели достигнуты",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не выводить детали, только summary",
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
    results = load_latest_result(str(results_path))

    # Проверяем метрики
    check_results = check_all_metrics(results)

    # Выводим результаты
    print_results(check_results, verbose=not args.quiet)

    # Сохраняем отчет
    if args.output:
        save_report(check_results, args.output)

    # Рекомендации
    if check_results["summary"]["critical"] > 0:
        print("\n💡 Рекомендации:")
        print("   1. Проверьте логи eval для деталей")
        print("   2. Проанализируйте failure cases")
        print("   3. Рассмотрите hyperparameter tuning")
        print("   4. Увеличьте размер датасета для более точной оценки")

    # Exit code
    if args.strict:
        if check_results["summary"]["achieved"] < check_results["summary"]["total"]:
            print("\n⚠️ Строгий режим: Не все целевые метрики достигнуты")
            sys.exit(1)

    if check_results["summary"]["critical"] > 0:
        print("\n⚠️ Обнаружены критичные метрики!")
        sys.exit(1)

    print("\n✅ Проверка пройдена!")
    sys.exit(0)


if __name__ == "__main__":
    main()
