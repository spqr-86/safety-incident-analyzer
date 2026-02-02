# 🚀 Quick Start: Evaluation & Metrics

Краткое руководство для быстрого старта работы с eval системой.

## 📋 Приоритеты (что делать в первую очередь)

### ✅ Этап 1: Запустить baseline оценку (1-2 дня)

**Цель:** Понять текущее качество системы

**Задачи:**
1. Запустить A/B тест на существующем датасете
2. Зафиксировать baseline метрики
3. Проанализировать результаты

**Команды:**
```bash
# 1. Убедитесь, что датасет существует
cat tests/dataset.csv

# 2. Настройте LangSmith (если еще не настроено)
export LANGSMITH_API_KEY="your_key"
export LANGSMITH_PROJECT="safety-incident-analyzer"

# 3. Запустите baseline eval
python run_ab_test.py

# 4. Проверьте результаты в LangSmith UI
# https://smith.langchain.com/
```

**Ожидаемые результаты:**
- Correctness score (0-10): целевое значение > 7.0
- QA score: целевое значение > 0.8
- Context QA score: целевое значение > 0.8

**Сохраните результаты:**
```bash
# Создайте файл с baseline метриками
cat > benchmarks/baseline.json << 'EOF'
{
  "date": "2025-12-14",
  "dataset": "golden-questions",
  "dataset_size": 16,
  "metrics": {
    "correctness_score": 7.5,
    "qa_score": 0.85,
    "context_qa_score": 0.82
  },
  "config": {
    "llm": "gigachat",
    "embeddings": "openai/text-embedding-3-small",
    "chunk_size": 1200,
    "vector_search_k": 10
  }
}
EOF
```

---

### ✅ Этап 2: Добавить retrieval метрики (2-3 дня)

**Цель:** Оценить качество поиска документов

**Задачи:**

1. **Создать скрипт для eval retrieval:**
```bash
# Создайте: eval/test_retrieval.py
```

```python
# eval/test_retrieval.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval_metrics import evaluate_retrieval_batch
from src.final_chain import create_final_hybrid_chain
import csv

def main():
    # Загружаем датасет
    dataset = []
    with open("tests/dataset.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dataset = list(reader)

    # Создаем retrieval компонент
    chain, retriever = create_final_hybrid_chain()

    # Для каждого вопроса получаем документы
    retrieved_docs_list = []
    for item in dataset:
        question = item["question"].replace('[cite_start]', '')
        docs = retriever.get_relevant_documents(question)

        # Извлекаем ID документов (или используем page_content для сравнения)
        doc_ids = [str(i) for i, _ in enumerate(docs)]  # упрощенно
        retrieved_docs_list.append(doc_ids)

    # Для начала - упрощенная оценка hit rate
    # (полноценная оценка требует аннотации релевантных документов)
    print(f"✅ Retrieval оценка завершена")
    print(f"📊 Извлечено документов для {len(dataset)} вопросов")
    print(f"Среднее количество документов: {sum(len(d) for d in retrieved_docs_list) / len(retrieved_docs_list):.1f}")

    # TODO: Добавить реальные метрики после аннотации релевантности

if __name__ == "__main__":
    main()
```

2. **Запустите:**
```bash
mkdir -p eval
python eval/test_retrieval.py
```

3. **Аннотируйте релевантность (частично, для начала):**
   - Выберите 10 вопросов из датасета
   - Для каждого вопроса посмотрите, какие документы retriever нашел
   - Отметьте, какие из них действительно релевантны
   - Создайте `tests/retrieval_sample.json`

---

### ✅ Этап 3: Расширить датасет до 50 вопросов (3-5 дней)

**Цель:** Увеличить надежность eval

**Задачи:**

1. **Генерация вопросов:**
```bash
# Запустите скрипт генерации
mkdir -p scripts
python scripts/generate_questions.py
```

2. **Ручная проверка:**
```bash
# Откройте сгенерированные вопросы
cat tests/dataset_extended.csv

# Проверьте качество:
# - Корректность вопросов
# - Корректность ответов
# - Соответствие контексту документов
```

3. **Объединение датасетов:**
```bash
# Создайте финальный датасет
cat tests/dataset.csv > tests/dataset_full.csv
tail -n +2 tests/dataset_extended.csv >> tests/dataset_full.csv

# Обновите run_ab_test.py
# DATASET_NAME = "golden-questions-full"
```

4. **Загрузите в LangSmith:**
```python
from langsmith import Client

client = Client()

# Создайте новый датасет в LangSmith из dataset_full.csv
# (используйте LangSmith UI или API)
```

---

### ✅ Этап 4: Добавить advanced generation метрики (2-3 дня)

**Цель:** Детальная оценка качества ответов

**Задачи:**

1. **Интеграция в run_ab_test.py:**
```python
# Добавьте в run_ab_test.py
from src.advanced_generation_metrics import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_citation_quality
)

# Создайте custom evaluator
def advanced_evaluator(run, example):
    question = example.inputs.get("question")
    answer = run.outputs.get("output")
    context = run.outputs.get("context", "")  # если доступен

    metrics = {}

    # Faithfulness
    faith_result = evaluate_faithfulness(question, context, answer, judge_llm)
    metrics.update(faith_result)

    # Answer relevance
    rel_result = evaluate_answer_relevance(question, answer, judge_llm)
    metrics.update(rel_result)

    # Citation quality
    cite_result = evaluate_citation_quality(answer, context, [])
    metrics.update(cite_result)

    # Возвращаем агрегированный score
    return {
        "score": metrics.get("faithfulness_score", 0.0),
        "comment": f"Faith: {metrics.get('faithfulness_score'):.2f}, Rel: {metrics.get('answer_relevance_score'):.2f}"
    }

# Добавьте в evaluation_config
evaluation_config = RunEvalConfig(
    custom_evaluators=[check_correctness, advanced_evaluator],
    evaluators=["qa", "context_qa"],
    eval_llm=judge_llm,
)
```

2. **Запустите extended eval:**
```bash
python run_ab_test.py
```

---

### ✅ Этап 5: Настроить CI/CD (2-3 дня)

**Цель:** Автоматический eval при изменениях

**Задачи:**

1. **Создайте GitHub Action:**
```yaml
# .github/workflows/evaluation.yml
name: Evaluation

on:
  pull_request:
  push:
    branches: [main, develop]
  workflow_dispatch:  # ручной запуск

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run evaluation
        env:
          GIGACHAT_CREDENTIALS: ${{ secrets.GIGACHAT_CREDENTIALS }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
        run: |
          python run_ab_test.py

      - name: Compare with baseline
        run: |
          python scripts/compare_metrics.py
```

2. **Создайте скрипт сравнения:**
```bash
# scripts/compare_metrics.py
# TODO: Реализовать сравнение текущих метрик с baseline
```

---

## 📊 Дашборд метрик

### Что отслеживать:

1. **Correctness** (главная метрика)
   - Целевое значение: > 8.0/10
   - Критичное значение: < 6.0 (алерт!)

2. **Faithfulness** (галлюцинации)
   - Целевое значение: > 0.9
   - Критичное значение: < 0.7

3. **Answer Relevance**
   - Целевое значение: > 0.85

4. **Retrieval Hit Rate @ 10**
   - Целевое значение: > 0.9

5. **Latency (P95)**
   - Целевое значение: < 10 секунд

6. **Cost per query**
   - Целевое значение: < $0.05

---

## 🐛 Troubleshooting

### Проблема: LangSmith не работает
```bash
# Проверьте переменные окружения
echo $LANGSMITH_API_KEY
echo $LANGSMITH_PROJECT

# Установите правильно
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_TRACING_V2=true
```

### Проблема: Низкий correctness score
1. Проверьте quality of retrieval (получены ли релевантные документы?)
2. Проверьте промпты в agents (RelevanceChecker, ResearchAgent)
3. Увеличьте `VECTOR_SEARCH_K` (попробуйте 15-20)
4. Попробуйте другие веса hybrid retrieval

### Проблема: Галлюцинации (низкий faithfulness)
1. Ужесточите промпт в ResearchAgent ("используй ТОЛЬКО контекст")
2. Добавьте citation requirement в промпт
3. Увеличьте temperature = 0.0 (детерминистичность)

---

## 📈 Следующие шаги

После завершения Quick Start:

1. **Расширить датасет до 100+ вопросов**
2. **Настроить W&B для experiment tracking**
3. **Добавить production мониторинг**
4. **Провести hyperparameter tuning**
5. **Настроить drift detection**

---

## 📚 Полезные ссылки

- [Roadmap проекта](../ROADMAP.md)
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [RAGAS Framework](https://docs.ragas.io/)
- [Retrieval метрики](./src/retrieval_metrics.py)
- [Advanced Generation метрики](./src/advanced_generation_metrics.py)

---

**Вопросы?** Создайте issue в репозитории.
