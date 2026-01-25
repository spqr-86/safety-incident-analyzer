"""
Скрипт для генерации дополнительных вопросов для тестового датасета.

Использует LLM для генерации синтетических вопросов из документов
и вариаций существующих вопросов.
"""

import os
import sys
import csv
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_factory import get_llm
from src.vector_store import load_vector_store
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()


def generate_questions_from_context(context: str, llm, num_questions: int = 3) -> List[Dict[str, str]]:
    """
    Генерирует вопросы и ответы из предоставленного контекста.

    Args:
        context: Текст документа/чанка
        llm: LLM для генерации
        num_questions: Количество вопросов для генерации

    Returns:
        Список словарей с вопросами и ответами
    """
    prompt = ChatPromptTemplate.from_template(
        """Ты - эксперт по созданию тестовых вопросов для системы вопрос-ответ по охране труда.

# ЗАДАЧА:
На основе предоставленного Контекста создай {num_questions} вопроса с эталонными ответами.

# ТРЕБОВАНИЯ:
1. Вопросы должны быть четкими и конкретными
2. Ответы должны строго соответствовать Контексту
3. Разнообразие: простые факты, перечисления, периодичность, условия и т.д.
4. Включи в ответ цитаты в формате [cite: X] (можно использовать X=999 как плейсхолдер)
5. Вопросы должны быть естественными (как спросил бы реальный пользователь)

# ФОРМАТ ОТВЕТА:
Верни JSON-массив объектов:
[
  {{
    "question": "Ваш вопрос?",
    "ground_truth": "Эталонный ответ с фактами из контекста [cite: 999].",
    "difficulty": 1,  # 1=простой, 2=средний, 3=сложный
    "category": "название категории"
  }},
  ...
]

# КОНТЕКСТ:
{context}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "num_questions": num_questions})

    try:
        questions = json.loads(response)
        return questions if isinstance(questions, list) else []
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
        print(f"Ответ LLM: {response}")
        return []


def generate_question_variations(
    original_question: str, original_answer: str, llm, num_variations: int = 2
) -> List[Dict[str, str]]:
    """
    Генерирует вариации существующего вопроса.

    Args:
        original_question: Исходный вопрос
        original_answer: Исходный ответ
        llm: LLM
        num_variations: Количество вариаций

    Returns:
        Список вариаций
    """
    prompt = ChatPromptTemplate.from_template(
        """Создай {num_variations} вариации вопроса с разными формулировками.

# ТРЕБОВАНИЯ:
1. Вопросы должны спрашивать о той же информации, но другими словами
2. Разная структура предложения (прямой вопрос, косвенный, с уточнением и т.д.)
3. Ответ должен остаться тем же (можно немного перефразировать для естественности)

# ОРИГИНАЛ:
Вопрос: {question}
Ответ: {answer}

# ФОРМАТ:
Верни JSON: [{{"question": "...", "ground_truth": "...", "difficulty": 1, "category": "variation"}}]

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "question": original_question,
        "answer": original_answer,
        "num_variations": num_variations,
    })

    try:
        variations = json.loads(response)
        return variations if isinstance(variations, list) else []
    except json.JSONDecodeError:
        print(f"Ошибка парсинга вариаций для вопроса: {original_question}")
        return []


def sample_documents_from_vectorstore(vector_store, num_samples: int = 10) -> List[str]:
    """
    Получает случайные документы из векторного хранилища.

    Args:
        vector_store: ChromaDB vector store
        num_samples: Количество документов

    Returns:
        Список текстов документов
    """
    # Простой запрос для получения документов
    results = vector_store.similarity_search(
        "охрана труда обучение инструктаж", k=num_samples
    )
    return [doc.page_content for doc in results]


def main():
    """
    Основная функция для генерации вопросов.
    """
    print("🚀 Запуск генерации вопросов...")

    # Инициализация
    llm = get_llm()
    vector_store = load_vector_store()

    # Читаем существующий датасет
    dataset_path = Path(__file__).parent.parent / "tests" / "dataset.csv"
    existing_questions = []

    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_questions = list(reader)
        print(f"📖 Загружено {len(existing_questions)} существующих вопросов")

    # Генерация новых вопросов
    new_questions = []

    # Стратегия 1: Вариации существующих вопросов
    print("\n🔄 Генерация вариаций существующих вопросов...")
    for i, q_data in enumerate(existing_questions[:5], 1):  # Берем первые 5 для примера
        print(f"  Обработка вопроса {i}/5...")
        variations = generate_question_variations(
            q_data["question"].replace('[cite_start]', ''),
            q_data["ground_truth"],
            llm,
            num_variations=2,
        )
        new_questions.extend(variations)
        print(f"    ✅ Сгенерировано {len(variations)} вариаций")

    # Стратегия 2: Генерация из документов
    print("\n📄 Генерация вопросов из документов...")
    documents = sample_documents_from_vectorstore(vector_store, num_samples=5)

    for i, doc in enumerate(documents, 1):
        print(f"  Обработка документа {i}/{len(documents)}...")
        # Берем фрагмент документа (первые 1000 символов для контекста)
        context = doc[:1000]
        questions = generate_questions_from_context(context, llm, num_questions=2)
        new_questions.extend(questions)
        print(f"    ✅ Сгенерировано {len(questions)} вопросов")

    # Сохранение в новый файл
    output_path = Path(__file__).parent.parent / "tests" / "dataset_extended.csv"

    print(f"\n💾 Сохранение {len(new_questions)} новых вопросов в {output_path}...")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        if new_questions:
            fieldnames = ["question", "ground_truth", "difficulty", "category"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_questions)

    print(f"✅ Готово! Сгенерировано {len(new_questions)} вопросов")
    print(f"📊 Статистика:")
    print(f"  - Вариаций существующих: ~{min(len(existing_questions) * 2, 10)}")
    print(f"  - Из документов: ~{len(documents) * 2}")
    print(f"\n💡 Следующие шаги:")
    print(f"  1. Проверьте сгенерированные вопросы в {output_path}")
    print(f"  2. Вручную отредактируйте/удалите некачественные вопросы")
    print(f"  3. Объедините с основным датасетом: cat tests/dataset.csv tests/dataset_extended.csv > tests/dataset_full.csv")
    print(f"  4. Обновите run_ab_test.py для использования нового датасета")


if __name__ == "__main__":
    main()
