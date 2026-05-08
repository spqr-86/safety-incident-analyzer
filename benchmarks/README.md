# Benchmarks и Baseline Метрики

Эта директория содержит baseline метрики и историю результатов eval.

## Файлы

### `baseline.json`
Baseline метрики для текущей production версии системы. Используется для сравнения новых версий.

**Актуальный конфиг (май 2026, V7 pipeline):**
```json
{
  "date": "2026-05-07",
  "version": "V7 (stage 6)",
  "dataset": "golden-questions",
  "dataset_size": 41,
  "config": {
    "pipeline": "v7_langgraph",
    "llm_provider": "gemini",
    "llm_model": "gemini-3-flash-preview",
    "thinking_budget": 4096,
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "vector_search_k": 40,
    "hybrid_weights": [0.5, 0.5],
    "evaluate_complex_top_k": 24
  },
  "metrics": {
    "note": "Baseline eval не запускался на V7. Требуется: python eval/run_v7_eval.py"
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
