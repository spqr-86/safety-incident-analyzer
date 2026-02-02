# 📚 Примеры использования Evaluation системы

Практические примеры работы с eval инфраструктурой.

---

## 🚀 Базовые сценарии

### 1. Первый запуск: Создание baseline

```bash
# Шаг 1: Запустить полную оценку
python eval/run_full_evaluation.py

# Шаг 2: Проверить результаты
cat benchmarks/results_history.jsonl | jq '.'

# Шаг 3: Создать baseline из последнего запуска
python -c "
import json

# Читаем последний результат
with open('benchmarks/results_history.jsonl', 'r') as f:
    lines = f.readlines()
    latest = json.loads(lines[-1])

# Создаем baseline
baseline = {
    'date': latest['timestamp'],
    'version': 'v1.0.0',
    'dataset': latest['dataset'],
    'dataset_size': latest['dataset_size'],
    'metrics': latest['aggregate_metrics']
}

# Сохраняем
with open('benchmarks/baseline.json', 'w') as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print('✅ Baseline создан!')
"
```

---

### 2. Быстрая проверка (3-5 вопросов)

```bash
# Для быстрой проверки после изменений
python eval/run_full_evaluation.py --limit 5

# Пример вывода:
# [1/5] Кто проходит обучение по программе А?...
#   ✅ Correctness: 8.5/10
#   📊 Faithfulness: 0.92
#   ⏱️  Время: 7.8s
```

---

### 3. Сравнение с baseline

```bash
# После запуска eval
python scripts/compare_with_baseline.py

# Пример вывода:
# ======================================================================
# 📊 СРАВНЕНИЕ С BASELINE
# ======================================================================
#
# 📅 Baseline дата: 2025-12-14
# 📅 Текущий запуск: 2025-12-15T10:30:00
#
# 🎯 Общий статус: ✅ IMPROVED
#    Улучшений: 2
#    Регрессий: 0
#    Стабильно: 3
#
# ✅ УЛУЧШЕНИЯ:
#    • Faithfulness
#      Baseline: 0.850
#      Текущее:  0.920
#      Изменение: +0.070 (+8.2%)
```

---

### 4. A/B тестирование разных конфигураций

```bash
# Тест 1: Текущая конфигурация
python eval/run_full_evaluation.py --output benchmarks/test_config_a.jsonl

# Изменить параметры в config/settings.py
# Например: VECTOR_SEARCH_K = 15

# Тест 2: Новая конфигурация
python eval/run_full_evaluation.py --output benchmarks/test_config_b.jsonl

# Сравнить результаты
python -c "
import json

def load_latest(path):
    with open(path) as f:
        return json.loads(f.readlines()[-1])

config_a = load_latest('benchmarks/test_config_a.jsonl')
config_b = load_latest('benchmarks/test_config_b.jsonl')

print('Конфигурация A:')
print(f\"  Correctness: {config_a['aggregate_metrics']['mean_correctness_score']:.2f}\")
print(f\"  Latency: {config_a['aggregate_metrics']['mean_total_time']:.2f}s\")

print('\nКонфигурация B:')
print(f\"  Correctness: {config_b['aggregate_metrics']['mean_correctness_score']:.2f}\")
print(f\"  Latency: {config_b['aggregate_metrics']['mean_total_time']:.2f}s\")
"
```

---

## 🔬 Продвинутые сценарии

### 5. Eval с использованием LangSmith

```bash
# Установить переменные окружения
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_TRACING_V2=true
export LANGSMITH_PROJECT="safety-incident-analyzer"

# Запустить eval через LangSmith
python run_ab_test.py

# Результаты будут доступны в LangSmith UI:
# https://smith.langchain.com/
```

---

### 6. Генерация дополнительных тестовых вопросов

```bash
# Генерация вопросов из документов
python scripts/generate_questions.py

# Результат: tests/dataset_extended.csv
cat tests/dataset_extended.csv

# Проверить качество сгенерированных вопросов
head -n 5 tests/dataset_extended.csv | column -t -s','

# Объединить с основным датасетом
cat tests/dataset.csv > tests/dataset_full.csv
tail -n +2 tests/dataset_extended.csv >> tests/dataset_full.csv

echo "✅ Датасет расширен до $(wc -l < tests/dataset_full.csv) вопросов"
```

---

### 7. Мониторинг production метрик

```python
# Добавить в app.py для логирования запросов
import json
from datetime import datetime

def log_query_metrics(question, answer, retrieval_time, generation_time):
    """Логирует метрики запроса в production."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "question_length": len(question),
        "answer_length": len(answer),
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
    }

    with open("production_logs/queries.jsonl", "a") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
```

```bash
# Анализ production логов
python -c "
import json
import statistics

with open('production_logs/queries.jsonl') as f:
    metrics = [json.loads(line) for line in f]

latencies = [m['total_time'] for m in metrics]
print(f'Всего запросов: {len(metrics)}')
print(f'Средняя задержка: {statistics.mean(latencies):.2f}s')
print(f'P95 задержка: {statistics.quantiles(latencies, n=20)[18]:.2f}s')
"
```

---

### 8. CI/CD интеграция

```bash
# Локальная симуляция CI workflow
act -j evaluate  # Требует 'act' (https://github.com/nektos/act)

# Или запустить напрямую то, что делает CI:
python eval/run_full_evaluation.py --limit 5
python scripts/compare_with_baseline.py
```

---

### 9. Анализ конкретных failure cases

```python
# scripts/analyze_failures.py
import json

# Загружаем последние результаты
with open('benchmarks/results_history.jsonl') as f:
    latest = json.loads(f.readlines()[-1])

# Находим плохие результаты
failures = []
for result in latest['detailed_results']:
    if result.get('correctness_score', 10) < 6.0:
        failures.append({
            'question': result['question'],
            'answer': result['answer'],
            'ground_truth': result['ground_truth'],
            'correctness': result.get('correctness_score', 0),
            'faithfulness': result.get('faithfulness_score', 0),
        })

print(f"Найдено {len(failures)} неудачных ответов:")
for i, f in enumerate(failures, 1):
    print(f"\n{i}. {f['question'][:60]}...")
    print(f"   Correctness: {f['correctness']:.1f}/10")
    print(f"   Faithfulness: {f['faithfulness']:.2f}")
```

---

### 10. Экспорт метрик для визуализации

```python
# scripts/export_metrics_csv.py
import json
import csv

# Читаем историю
with open('benchmarks/results_history.jsonl') as f:
    history = [json.loads(line) for line in f]

# Экспортируем в CSV
with open('benchmarks/metrics_timeline.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'correctness', 'faithfulness', 'latency_p95'])

    for run in history:
        metrics = run['aggregate_metrics']
        writer.writerow([
            run['timestamp'],
            metrics.get('mean_correctness_score', 0),
            metrics.get('mean_faithfulness_score', 0),
            metrics.get('p95_total_time', 0),
        ])

print("✅ Экспорт завершен: benchmarks/metrics_timeline.csv")
```

```bash
# Визуализация в gnuplot (опционально)
gnuplot <<EOF
set datafile separator ','
set xdata time
set timefmt "%Y-%m-%dT%H:%M:%S"
set format x "%m/%d"
set terminal png size 800,600
set output 'benchmarks/metrics_plot.png'
plot 'benchmarks/metrics_timeline.csv' using 1:2 with lines title 'Correctness'
EOF
```

---

## 🎯 Рекомендуемые workflow

### Daily Development

```bash
# 1. Внесли изменения в код
# 2. Быстрая проверка
python eval/run_full_evaluation.py --limit 5

# 3. Если всё OK - коммит
git add .
git commit -m "Improve prompt template"

# 4. CI автоматически запустит eval
```

### Weekly Review

```bash
# 1. Полная оценка
python eval/run_full_evaluation.py

# 2. Сравнение с baseline
python scripts/compare_with_baseline.py

# 3. Анализ трендов
python scripts/analyze_trends.py  # TODO: создать

# 4. Обновить baseline если есть улучшения
```

### Before Release

```bash
# 1. Полная оценка на всём датасете
python eval/run_full_evaluation.py

# 2. Проверка всех целевых метрик
python scripts/check_target_metrics.py  # TODO: создать

# 3. Обновление baseline
cp benchmarks/baseline.json benchmarks/baseline_v1.0.0.json
# Создать новый baseline из results_history.jsonl

# 4. Tag release
git tag -a v1.1.0 -m "Release v1.1.0 with improved metrics"
```

---

## 📖 Дополнительная информация

- [Roadmap проекта](../ROADMAP.md)
- [Quick Start](../guides/quick-start.md)
- [Benchmarks README](../../benchmarks/README.md)
