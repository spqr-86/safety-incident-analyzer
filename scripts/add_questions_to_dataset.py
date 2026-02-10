"""
Скрипт для добавления новых вопросов в датасет.

Два режима работы:
1. Интерактивный: добавляйте вопросы один за другим
2. Batch: загрузка из JSON файла

Использование:
    # Интерактивный режим
    python scripts/add_questions_to_dataset.py --interactive

    # Batch из JSON
    python scripts/add_questions_to_dataset.py --input new_questions.json

    # Просмотр текущего датасета
    python scripts/add_questions_to_dataset.py --show
"""

import csv
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Загружает существующий датасет."""
    questions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(
                {"question": row["question"], "ground_truth": row["ground_truth"]}
            )
    return questions


def save_dataset(dataset_path: str, questions: List[Dict[str, str]]) -> None:
    """Сохраняет датасет в CSV."""
    with open(dataset_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth"])
        writer.writeheader()
        writer.writerows(questions)


def show_dataset(dataset_path: str) -> None:
    """Показывает текущий датасет."""
    questions = load_dataset(dataset_path)

    print("\n" + "=" * 70)
    print(f"📊 ТЕКУЩИЙ ДАТАСЕТ ({len(questions)} вопросов)")
    print("=" * 70)

    for i, item in enumerate(questions, 1):
        print(f"\n{i}. {item['question']}")
        print(f"   Ответ: {item['ground_truth'][:100]}...")

    print("\n" + "=" * 70)


def interactive_mode(dataset_path: str) -> None:
    """Интерактивное добавление вопросов."""
    print("\n" + "=" * 70)
    print("📝 ИНТЕРАКТИВНЫЙ РЕЖИМ ДОБАВЛЕНИЯ ВОПРОСОВ")
    print("=" * 70)
    print("\nИнструкция:")
    print("  1. Введите вопрос")
    print("  2. Введите эталонный ответ (с цитатами [cite: X, Y] если есть)")
    print("  3. Введите пустую строку в вопросе для завершения")
    print("\n" + "=" * 70)

    questions = load_dataset(dataset_path)
    initial_count = len(questions)

    while True:
        print("\n")
        question = input("❓ Вопрос: ").strip()

        if not question:
            break

        print(
            "💡 Эталонный ответ (можно вводить многострочно, двойной Enter для завершения):"
        )
        ground_truth_lines = []
        empty_count = 0

        while True:
            line = input()
            if not line.strip():
                empty_count += 1
                if empty_count >= 1:  # Один Enter для завершения
                    break
            else:
                empty_count = 0
                ground_truth_lines.append(line)

        ground_truth = " ".join(ground_truth_lines).strip()

        if not ground_truth:
            print("⚠️  Ответ пустой, вопрос пропущен")
            continue

        questions.append({"question": question, "ground_truth": ground_truth})

        print(f"✅ Добавлено (всего вопросов: {len(questions)})")

    if len(questions) > initial_count:
        save_dataset(dataset_path, questions)
        print(f"\n💾 Сохранено {len(questions) - initial_count} новых вопросов")
        print(f"   Всего в датасете: {len(questions)}")
    else:
        print("\n⚠️  Новые вопросы не добавлены")


def batch_mode(dataset_path: str, input_file: str) -> None:
    """Добавление вопросов из JSON файла."""
    print(f"\n📂 Загрузка вопросов из {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        new_questions = json.load(f)

    if not isinstance(new_questions, list):
        print("❌ JSON должен содержать массив объектов")
        sys.exit(1)

    # Валидация
    valid_questions = []
    for i, item in enumerate(new_questions):
        if not isinstance(item, dict):
            print(f"⚠️  Пропущен элемент {i}: не является объектом")
            continue

        if "question" not in item or "ground_truth" not in item:
            print(f"⚠️  Пропущен элемент {i}: отсутствует question или ground_truth")
            continue

        valid_questions.append(
            {
                "question": item["question"].strip(),
                "ground_truth": item["ground_truth"].strip(),
            }
        )

    print(f"✅ Валидировано {len(valid_questions)} вопросов")

    # Загружаем существующий датасет
    questions = load_dataset(dataset_path)
    initial_count = len(questions)

    # Добавляем новые
    questions.extend(valid_questions)

    # Сохраняем
    save_dataset(dataset_path, questions)

    print(f"💾 Добавлено {len(valid_questions)} новых вопросов")
    print(f"   Всего в датасете: {len(questions)}")


def create_template_json(output_path: str) -> None:
    """Создает шаблон JSON для batch режима."""
    template = [
        {
            "question": "Кто должен проходить обучение по охране труда?",
            "ground_truth": "Все работники организации, включая руководителя [cite: 219]. Обучение проводится по программам, разработанным с учетом специфики деятельности [cite: 220].",
        },
        {
            "question": "Какова периодичность проверки знаний по охране труда?",
            "ground_truth": "Периодическая проверка проводится не реже одного раза в год [cite: 225]. Внеплановая проверка проводится при изменении нормативных актов или условий труда [cite: 226].",
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"✅ Создан шаблон: {output_path}")
    print("\nФормат:")
    print(json.dumps(template[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Добавление вопросов в датасет")
    parser.add_argument(
        "--dataset",
        default="tests/dataset.csv",
        help="Путь к датасету (по умолчанию: tests/dataset.csv)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Интерактивный режим добавления вопросов",
    )
    parser.add_argument(
        "--input",
        help="JSON файл с новыми вопросами (batch режим)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показать текущий датасет",
    )
    parser.add_argument(
        "--create-template",
        help="Создать шаблон JSON файла",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # Проверяем существование датасета
    if not dataset_path.exists() and not args.create_template:
        print(f"❌ Датасет не найден: {dataset_path}")
        sys.exit(1)

    # Режимы работы
    if args.create_template:
        create_template_json(args.create_template)
    elif args.show:
        show_dataset(str(dataset_path))
    elif args.interactive:
        interactive_mode(str(dataset_path))
    elif args.input:
        batch_mode(str(dataset_path), args.input)
    else:
        print(
            "❌ Укажите режим работы: --interactive, --input, --show, или --create-template"
        )
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
