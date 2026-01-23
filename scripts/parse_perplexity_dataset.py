"""
Парсер для конвертации золотого датасета от Perplexity в CSV формат.
Извлекает вопросы и ответы из markdown структуры.
"""

import csv
import re
from pathlib import Path


def parse_perplexity_dataset(markdown_text: str) -> list[dict]:
    """
    Парсит markdown dataset от Perplexity и возвращает список вопрос-ответ пар.
    """
    questions = []

    # Регулярное выражение для поиска вопросов (### Q1: ... ### Q25:)
    # Ищем паттерн: ### Q{number}: {question}
    question_pattern = r'### Q(\d+): (.+?)\n'

    # Находим все вопросы
    question_matches = list(re.finditer(question_pattern, markdown_text))

    for i, match in enumerate(question_matches):
        q_number = match.group(1)
        question_text = match.group(2).strip()

        # Ищем ответ между текущим вопросом и следующим
        start_pos = match.end()

        # Определяем конец блока вопроса (следующий вопрос или конец уровня)
        if i + 1 < len(question_matches):
            end_pos = question_matches[i + 1].start()
        else:
            # Последний вопрос - ищем конец уровня или секции
            next_level = markdown_text.find('## УРОВЕНЬ', start_pos)
            meta_section = markdown_text.find('## МЕТАИНФОРМАЦИЯ', start_pos)
            end_pos = min(x for x in [next_level, meta_section, len(markdown_text)] if x > start_pos)

        block = markdown_text[start_pos:end_pos]

        # Извлекаем ответ (после **Ответ**: или **Ответ:**)
        answer_match = re.search(r'\*\*Ответ\*\*:\s*(.+?)(?=\n---|\n\n###|\Z)', block, re.DOTALL)

        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Очищаем ответ от лишних переносов и форматирования
            answer_text = re.sub(r'\n+', ' ', answer_text)
            answer_text = re.sub(r'\s+', ' ', answer_text)

            questions.append({
                'question': question_text,
                'ground_truth': answer_text
            })
            print(f"✅ Q{q_number}: Извлечен вопрос и ответ")
        else:
            print(f"⚠️  Q{q_number}: Ответ не найден")

    return questions


def load_existing_dataset(csv_path: str) -> list[dict]:
    """Загружает существующий CSV dataset."""
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'question': row['question'],
                'ground_truth': row['ground_truth']
            })
    return questions


def check_duplicates(new_questions: list[dict], existing_questions: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Проверяет на дубликаты по тексту вопроса (нечеткое сравнение).
    Возвращает уникальные вопросы и список дубликатов.
    """
    existing_questions_text = [q['question'].lower().strip() for q in existing_questions]

    unique_questions = []
    duplicates = []

    for new_q in new_questions:
        q_text_lower = new_q['question'].lower().strip()

        # Нечеткое сравнение (первые 50 символов)
        is_duplicate = False
        for existing_q_text in existing_questions_text:
            if q_text_lower[:50] == existing_q_text[:50]:
                is_duplicate = True
                duplicates.append(new_q['question'][:80] + '...')
                break

        if not is_duplicate:
            unique_questions.append(new_q)

    return unique_questions, duplicates


def save_merged_dataset(csv_path: str, all_questions: list[dict]):
    """Сохраняет объединенный dataset."""
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'ground_truth'])
        writer.writeheader()
        writer.writerows(all_questions)


def main():
    # Путь к markdown файлу от Perplexity
    markdown_file = Path('perplexity_golden_dataset.md')

    if not markdown_file.exists():
        print(f"❌ Файл {markdown_file} не найден")
        print("\nИнструкция:")
        print("1. Сохраните dataset от Perplexity в файл 'perplexity_golden_dataset.md'")
        print("2. Запустите скрипт снова: python scripts/parse_perplexity_dataset.py")
        return

    # Читаем markdown
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    print("🔍 Парсинг золотого датасета от Perplexity...")
    print("=" * 70)

    # Парсим вопросы
    new_questions = parse_perplexity_dataset(markdown_text)

    print("\n" + "=" * 70)
    print(f"📊 Извлечено вопросов: {len(new_questions)}")

    # Загружаем существующий dataset
    existing_csv = Path('tests/dataset.csv')
    if existing_csv.exists():
        existing_questions = load_existing_dataset(str(existing_csv))
        print(f"📁 Существующих вопросов: {len(existing_questions)}")

        # Проверяем дубликаты
        print("\n🔎 Проверка на дубликаты...")
        unique_questions, duplicates = check_duplicates(new_questions, existing_questions)

        if duplicates:
            print(f"⚠️  Найдено дубликатов: {len(duplicates)}")
            for dup in duplicates[:5]:
                print(f"   - {dup}")
            if len(duplicates) > 5:
                print(f"   ... и еще {len(duplicates) - 5}")
        else:
            print("✅ Дубликатов не найдено")

        # Объединяем
        all_questions = existing_questions + unique_questions
    else:
        print("⚠️  Существующий dataset не найден, создаем новый")
        all_questions = new_questions

    # Сохраняем
    print(f"\n💾 Сохранение объединенного dataset ({len(all_questions)} вопросов)...")
    save_merged_dataset(str(existing_csv), all_questions)

    print("\n" + "=" * 70)
    print("✅ ГОТОВО!")
    print(f"   Всего вопросов в dataset: {len(all_questions)}")
    print(f"   Новых добавлено: {len(new_questions) - len(duplicates) if existing_csv.exists() else len(new_questions)}")
    print(f"   Файл: {existing_csv}")
    print("=" * 70)

    # Показываем пример
    if all_questions:
        print("\n📝 Пример первого нового вопроса:")
        if existing_csv.exists():
            example = unique_questions[0] if unique_questions else new_questions[0]
        else:
            example = all_questions[0]
        print(f"   Q: {example['question'][:100]}...")
        print(f"   A: {example['ground_truth'][:150]}...")


if __name__ == '__main__':
    main()
