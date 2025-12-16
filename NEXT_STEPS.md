# 🚀 Следующие шаги - Action Plan

**Дата создания:** 2025-12-16
**Текущий статус:** Sprint 1 ~90% завершен

---

## ✅ Что уже готово

- ✅ Все 11 метрик реализованы и протестированы
- ✅ 6 utility scripts
- ✅ CI/CD pipeline настроен
- ✅ Документация (5 документов)
- ✅ Unit тесты пройдены
- ✅ 4 коммита запушены

---

## 🎯 Фаза 1: Запуск и валидация (1-2 дня)

### ШАГ 1.1: Первый запуск baseline evaluation

**Цель:** Получить первые реальные метрики системы

```bash
# Проверить, что все зависимости установлены
pip install -r requirements.txt

# Проверить наличие API ключей
echo $GIGACHAT_CREDENTIALS
echo $OPENAI_API_KEY

# Запустить eval на 5 вопросах (тест)
python eval/run_full_evaluation.py --limit 5

# Если OK - полная оценка на всём датасете (16 вопросов)
python eval/run_full_evaluation.py
```

**Ожидаемый результат:**
- Файл `benchmarks/results_history.jsonl` создан
- Получены первые метрики

**Время:** ~30-60 минут (зависит от API)

---

### ШАГ 1.2: Создание baseline

**Цель:** Зафиксировать текущие метрики как baseline

```bash
# Автоматически создать baseline из последнего запуска
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
    'config': {
        'llm_provider': 'gigachat',
        'embedding_provider': 'openai',
        'chunk_size': 1200,
        'vector_search_k': 10,
    },
    'metrics': latest['aggregate_metrics']
}

# Сохраняем
with open('benchmarks/baseline.json', 'w') as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print('✅ Baseline создан!')
"

# Проверить baseline
cat benchmarks/baseline.json | jq '.metrics'
```

**Ожидаемый результат:**
- Файл `benchmarks/baseline.json` создан
- Baseline метрики зафиксированы

---

### ШАГ 1.3: Проверка целевых метрик

```bash
# Проверить соответствие целям
python scripts/check_target_metrics.py

# Если есть критичные метрики - записать для анализа
python scripts/check_target_metrics.py --output benchmarks/initial_check.json
```

**Ожидаемый результат:**
- Понимание, какие метрики требуют улучшения
- Отчет в `benchmarks/initial_check.json`

---

### ШАГ 1.4: Коммит baseline

```bash
git add benchmarks/baseline.json benchmarks/results_history.jsonl
git commit -m "Add initial baseline metrics

Initial evaluation results:
- Dataset: golden-questions (16 items)
- Correctness: X.X/10
- Faithfulness: X.XX
- Answer Relevance: X.XX
- Citation Rate: X.XX
- P95 Latency: X.Xs

Next: Expand dataset and improve metrics
"

git push
```

---

## 🎯 Фаза 2: Расширение датасета (3-5 дней)

### ШАГ 2.1: Генерация дополнительных вопросов

```bash
# Сгенерировать вопросы из документов
python scripts/generate_questions.py

# Проверить результаты
wc -l tests/dataset_extended.csv
head -5 tests/dataset_extended.csv
```

**Цель:** Добавить 30-50 новых вопросов

---

### ШАГ 2.2: Ручная валидация и редактирование

1. Открыть `tests/dataset_extended.csv`
2. Проверить каждый вопрос:
   - ✅ Вопрос корректный и естественный?
   - ✅ Ответ полный и точный?
   - ✅ Цитаты соответствуют источникам?
3. Отредактировать/удалить некачественные
4. Добавить категории и сложность

**Формат:**
```csv
question,ground_truth,category,difficulty
"Вопрос?","Ответ [cite: X]","category_name",1
```

---

### ШАГ 2.3: Объединение датасетов

```bash
# Создать полный датасет
cat tests/dataset.csv > tests/dataset_full.csv
tail -n +2 tests/dataset_extended.csv >> tests/dataset_full.csv

# Проверить
wc -l tests/dataset_full.csv
echo "Всего вопросов: $(($(wc -l < tests/dataset_full.csv) - 1))"
```

**Цель:** Минимум 50 вопросов в итоговом датасете

---

### ШАГ 2.4: Повторная оценка на расширенном датасете

```bash
# Временно изменить путь к датасету в eval/run_full_evaluation.py
# или использовать с --dataset параметром (если добавим)

# Запустить полную оценку
python eval/run_full_evaluation.py

# Сравнить с baseline
python scripts/compare_with_baseline.py

# Если метрики стабильны - обновить baseline
```

---

## 🎯 Фаза 3: Активация CI/CD (1 день)

### ШАГ 3.1: Настройка GitHub Secrets

В настройках репозитория на GitHub:

1. Settings → Secrets and variables → Actions
2. Добавить secrets:
   - `GIGACHAT_CREDENTIALS`
   - `OPENAI_API_KEY`
   - `LANGSMITH_API_KEY`

**Опционально (для drift alerts):**
   - `ALERT_EMAIL_USERNAME`
   - `ALERT_EMAIL_PASSWORD`
   - `ALERT_EMAIL_TO`

---

### ШАГ 3.2: Тестовый запуск CI/CD

```bash
# Создать тестовую ветку
git checkout -b test/ci-evaluation

# Сделать небольшое изменение
echo "# CI/CD Test" >> README.md

# Коммит и push
git add README.md
git commit -m "test: trigger CI/CD evaluation"
git push -u origin test/ci-evaluation

# Открыть PR и проверить, что workflow запустился
# https://github.com/spqr-86/safety-incident-analyzer/pulls
```

**Ожидается:**
- ✅ Workflow запустился
- ✅ Eval прошел на 5 вопросах
- ✅ Комментарий в PR с результатами

---

### ШАГ 3.3: Merge в main

```bash
# После успешного теста - merge главной ветки
git checkout main
git merge claude/project-development-plan-Wf6gO
git push

# Теперь CI/CD активен для всех PR!
```

---

## 🎯 Фаза 4: Sprint 2 - Experiment Tracking (1-2 недели)

### ШАГ 4.1: Настройка Weights & Biases (W&B)

```bash
# Установить W&B
pip install wandb

# Логин
wandb login

# Добавить в .env
echo "WANDB_API_KEY=your_key" >> .env
echo "WANDB_PROJECT=safety-incident-analyzer" >> .env
```

---

### ШАГ 4.2: Интеграция W&B в eval

Создать `src/experiment_tracker.py`:

```python
import wandb
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, project_name: str, run_name: str = None):
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                "chunk_size": 1200,
                "chunk_overlap": 150,
                "vector_search_k": 10,
                # ... другие параметры
            }
        )

    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
```

Обновить `eval/run_full_evaluation.py`:

```python
from src.experiment_tracker import ExperimentTracker

# В начале main()
tracker = ExperimentTracker("safety-incident-analyzer", run_name="eval-run")

# После получения метрик
tracker.log_metrics(agg_metrics)
tracker.finish()
```

---

### ШАГ 4.3: Hyperparameter Optimization

Создать `scripts/hyperparameter_search.py`:

```python
import optuna
# ... реализация поиска оптимальных параметров
```

**Параметры для оптимизации:**
- `VECTOR_SEARCH_K`: [5, 10, 15, 20]
- `HYBRID_WEIGHTS`: различные комбинации
- `CHUNK_SIZE`: [800, 1000, 1200, 1500]
- `TEMPERATURE`: [0.0, 0.1, 0.3, 0.5]

**Цель:** Найти оптимальную конфигурацию

---

## 🎯 Фаза 5: Production Monitoring (1-2 недели)

### ШАГ 5.1: Добавить логирование в app.py

```python
import json
from datetime import datetime

def log_query(question, answer, retrieval_time, generation_time):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
    }

    with open("production_logs/queries.jsonl", "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
```

---

### ШАГ 5.2: Weekly drift detection

Уже настроено в CI/CD! Каждое воскресенье автоматически:
- ✅ Запуск полной оценки
- ✅ Сравнение с baseline
- ✅ Email alert если есть drift

---

### ШАГ 5.3: Dashboard (опционально)

Создать Streamlit dashboard для мониторинга:

```bash
# Новая страница в app.py или отдельный файл
# monitoring_dashboard.py
```

---

## 📊 Метрики успеха по фазам

### Фаза 1 (Baseline):
- ✅ Baseline создан
- ✅ Первые метрики получены
- ✅ Идентифицированы слабые места

### Фаза 2 (Датасет):
- ✅ Датасет расширен до 50+ вопросов
- ✅ Метрики стабильны на большем датасете
- ✅ Разнообразие вопросов по темам

### Фаза 3 (CI/CD):
- ✅ CI/CD активен
- ✅ PR автоматически проверяются
- ✅ Drift detection работает

### Фаза 4 (Optimization):
- ✅ W&B интегрирован
- ✅ Найдены оптимальные параметры
- ✅ Метрики улучшены на 5-10%

### Фаза 5 (Production):
- ✅ Production логирование работает
- ✅ Weekly drift reports
- ✅ Dashboard для мониторинга

---

## 🚨 Риски и митигации

### Риск 1: API лимиты/стоимость
**Митигация:**
- Использовать `--limit` для быстрых проверок
- Кэшировать результаты
- Оптимизировать количество API вызовов

### Риск 2: Недостаточно данных для eval
**Митигация:**
- Приоритет: расширение датасета
- Использовать synthetic questions
- Собирать real user questions из production

### Риск 3: Метрики не улучшаются
**Митигация:**
- Error analysis для понимания проблем
- Systematic hyperparameter tuning
- A/B тестирование разных подходов

---

## 📅 Timeline

```
Неделя 1: Фаза 1-2 (Baseline + Датасет)
├── День 1-2: Запуск baseline eval
├── День 3-4: Генерация вопросов
└── День 5-7: Валидация и расширение датасета

Неделя 2: Фаза 3 (CI/CD активация)
├── День 8-9: Настройка GitHub Secrets
├── День 10: Тестовый PR
└── День 11-14: Merge и мониторинг

Неделя 3-4: Фаза 4 (Optimization)
├── W&B setup
├── Hyperparameter search
└── Применение оптимизаций

Месяц 2: Фаза 5 (Production)
├── Production monitoring
├── Drift detection
└── Continuous improvement
```

---

## ✅ Чек-лист немедленных действий

**СЕГОДНЯ:**
- [ ] Запустить `python eval/run_full_evaluation.py --limit 5`
- [ ] Если работает → полная оценка на 16 вопросах
- [ ] Создать baseline.json
- [ ] Проверить целевые метрики
- [ ] Закоммитить baseline

**ЭТА НЕДЕЛЯ:**
- [ ] Сгенерировать 30+ вопросов
- [ ] Валидировать вручную
- [ ] Создать dataset_full.csv (50+ вопросов)
- [ ] Повторная оценка на расширенном датасете
- [ ] Настроить GitHub Secrets

**СЛЕДУЮЩАЯ НЕДЕЛЯ:**
- [ ] Тестовый PR с CI/CD
- [ ] Merge в main
- [ ] Начать Sprint 2 (W&B)

---

**🎯 Главное правило:** Делать по одному шагу за раз, тестировать, коммитить.

**Вопросы?** См. документацию или создайте issue.
