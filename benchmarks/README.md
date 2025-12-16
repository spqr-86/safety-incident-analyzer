# Benchmarks и Baseline Метрики

Эта директория содержит baseline метрики и историю результатов eval.

## Файлы

### `baseline.json`
Baseline метрики для текущей production версии системы. Используется для сравнения новых версий.

**Формат:**
```json
{
  "date": "2025-12-14",
  "version": "v1.0.0",
  "dataset": "golden-questions",
  "dataset_size": 16,
  "config": {
    "llm_provider": "gigachat",
    "llm_model": "GigaChat",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "vector_search_k": 10,
    "hybrid_weights": [0.6, 0.4]
  },
  "metrics": {
    "correctness_score": 7.5,
    "faithfulness_score": 0.85,
    "answer_relevance_score": 0.82,
    "citation_rate": 0.95,
    "mean_total_time": 8.5,
    "p95_total_time": 12.0
  }
}
```

### `results_history.jsonl`
История всех запусков eval в формате JSONL (одна строка = один запуск).

Каждая запись содержит:
- timestamp
- dataset
- aggregate_metrics
- detailed_results (опционально)

## Как обновить baseline

После значительного улучшения системы:

```bash
# 1. Запустить полную оценку
python eval/run_full_evaluation.py

# 2. Если метрики улучшились - обновить baseline
cp benchmarks/baseline.json benchmarks/baseline_old.json
# Создать новый baseline.json с новыми метриками
```

## Целевые метрики

| Метрика | Целевое | Baseline | Статус |
|---------|---------|----------|--------|
| Correctness | > 8.0/10 | 7.5 | 🔄 Требуется улучшение |
| Faithfulness | > 0.90 | 0.85 | 🔄 Требуется улучшение |
| Answer Relevance | > 0.85 | 0.82 | 🔄 Требуется улучшение |
| Citation Rate | > 0.95 | 0.95 | ✅ Достигнуто |
| P95 Latency | < 10s | 12.0s | 🔄 Требуется улучшение |

## Сравнение с baseline

```bash
# Запустить скрипт сравнения (когда будет реализован)
python scripts/compare_with_baseline.py benchmarks/results_history.jsonl
```
