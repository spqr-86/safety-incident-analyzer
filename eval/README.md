# Evaluation Scripts

Эта директория содержит скрипты для оценки качества RAG системы.

## Структура

```
eval/
├── README.md                    # Этот файл
├── test_retrieval.py           # Оценка качества retrieval
├── test_generation.py          # Оценка качества generation
└── run_full_evaluation.py      # Полная оценка всей системы
```

## Использование

### Baseline оценка
```bash
# Запустить полную оценку
python eval/run_full_evaluation.py

# Только retrieval
python eval/test_retrieval.py

# Только generation
python eval/test_generation.py
```

### Интеграция с CI/CD
См. `.github/workflows/evaluation.yml`

## Метрики

### Retrieval метрики
- Hit Rate @ K
- MRR (Mean Reciprocal Rank)
- NDCG @ K
- Precision @ K
- Recall @ K

### Generation метрики
- Correctness (LLM-as-judge)
- Faithfulness (groundedness)
- Answer Relevance
- Context Relevance
- Completeness
- Citation Quality

### Pipeline метрики
- Latency (retrieval, generation, total)
- Cost (API calls, tokens)
- Answer length statistics

## Датасеты

- `tests/dataset.csv` - baseline golden questions (16 items)
- `tests/dataset_extended.csv` - сгенерированные вопросы
- `tests/dataset_full.csv` - объединенный датасет
- `tests/retrieval_dataset.jsonl` - аннотированные релевантные чанки

## Результаты

Результаты сохраняются в:
- `benchmarks/baseline.json` - baseline метрики
- `benchmarks/results_history.jsonl` - история всех запусков
- LangSmith - детальные трейсы и метрики

## См. также

- [План развития](../DEVELOPMENT_PLAN.md)
- [Quick Start](../QUICK_START.md)
