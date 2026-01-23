# 🚀 Локальная установка и запуск проекта

## 📥 Шаг 1: Клонирование проекта

```bash
# Клонируйте репозиторий
git clone https://github.com/spqr-86/safety-incident-analyzer.git
cd safety-incident-analyzer

# Проверьте текущую ветку
git branch -a
```

---

## 🔄 Шаг 2: Merge в main (если работали в feature ветке)

```bash
# Переключитесь на main
git checkout main

# Подтяните последние изменения
git pull origin main

# Смержите вашу feature ветку
git merge feature/eval-system

# Или если ветка называется claude/project-development-plan-Wf6gO:
# git merge claude/project-development-plan-Wf6gO

# Отправьте изменения в main
git push origin main
```

**Альтернатива через Pull Request** (рекомендуется для production):
```bash
# Создайте PR через GitHub UI
# Перейдите на https://github.com/spqr-86/safety-incident-analyzer
# Нажмите "Compare & pull request"
# Укажите base: main, compare: feature/eval-system
# Создайте PR и смержите через UI
```

---

## 🐍 Шаг 3: Настройка Python окружения

```bash
# Создайте виртуальное окружение
python3 -m venv venv

# Активируйте окружение
# На Linux/Mac:
source venv/bin/activate
# На Windows:
# venv\Scripts\activate

# Обновите pip
pip install --upgrade pip

# Установите зависимости
pip install -r requirements.txt
```

---

## 🔐 Шаг 4: Настройка API ключей

```bash
# Создайте .env файл из шаблона
cp .env.example .env

# Отредактируйте .env и добавьте ваши ключи
nano .env  # или любой другой редактор
```

**Минимальная конфигурация для OpenAI:**
```env
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.0

OPENAI_API_KEY=sk-ваш-ключ-здесь

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

**Или для GigaChat + OpenAI:**
```env
LLM_PROVIDER=gigachat
MODEL_NAME=GigaChat
TEMPERATURE=0.0

GIGACHAT_CREDENTIALS=ваш-gigachat-токен
OPENAI_API_KEY=sk-ваш-openai-ключ

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

---

## ✅ Шаг 5: Запуск тестов

### 5.1 Unit тесты (быстрые, без API)

```bash
# Запуск всех unit тестов
pytest tests/test_retrieval_metrics.py -v

# Запуск с покрытием кода
pytest tests/ --cov=src --cov-report=html

# Посмотреть результаты покрытия
open htmlcov/index.html  # или просто откройте в браузере
```

**Ожидаемый вывод:**
```
tests/test_retrieval_metrics.py::TestHitRate::test_hit_found PASSED
tests/test_retrieval_metrics.py::TestHitRate::test_hit_not_found PASSED
tests/test_retrieval_metrics.py::TestMRR::test_mrr_first_position PASSED
...
==================== 15 passed in 0.5s ====================
```

### 5.2 Демо метрик (без API ключей)

```bash
# Демонстрация работы метрик
python scripts/demo_metrics.py
```

### 5.3 Evaluation тесты (требуют API ключи)

```bash
# Быстрая оценка на 5 вопросах
python eval/run_full_evaluation.py --limit 5

# Полная оценка на всех 41 вопросах
python eval/run_full_evaluation.py

# Сравнение с baseline
python scripts/compare_with_baseline.py

# Проверка целевых метрик
python scripts/check_target_metrics.py
```

### 5.4 A/B тестирование (LangSmith)

```bash
# Добавьте в .env:
# LANGSMITH_API_KEY=ваш-ключ
# LANGSMITH_TRACING_V2=true
# LANGSMITH_PROJECT=safety-incident-analyzer

# Запустите A/B тест
python run_ab_test.py
```

---

## 🚀 Шаг 6: Запуск приложения

### 6.1 Индексация документов

```bash
# Положите ваши документы в source_docs/
mkdir -p source_docs
# Скопируйте .pdf, .docx, .md, .txt файлы

# Запустите индексацию
python index.py
```

**Ожидаемый вывод:**
```
📄 Обработка: document1.pdf
✅ Создано 45 чанков
📄 Обработка: document2.docx
✅ Создано 32 чанка
💾 Всего индексировано: 77 документов
```

### 6.2 Запуск Streamlit UI

```bash
# Запустите веб-приложение
streamlit run app.py
```

Откроется в браузере: http://localhost:8501

---

## 🧪 Полный чеклист тестирования

```bash
# 1. Unit тесты
pytest tests/test_retrieval_metrics.py -v
# ✅ Все тесты должны пройти

# 2. Демо метрик
python scripts/demo_metrics.py
# ✅ Должны показаться примеры работы метрик

# 3. Быстрая оценка (требует API ключи)
python eval/run_full_evaluation.py --limit 5
# ✅ Должны появиться метрики: correctness, faithfulness, citation_rate

# 4. Индексация (требует документы)
python index.py
# ✅ Должна создаться папка chroma_db_*

# 5. Streamlit UI
streamlit run app.py
# ✅ Приложение должно открыться в браузере

# 6. Проверка baseline
python scripts/check_target_metrics.py
# ✅ Должны показаться результаты сравнения с целевыми метриками
```

---

## 📊 Создание baseline метрик (для резюме!)

```bash
# 1. Запустите полную оценку
python eval/run_full_evaluation.py

# 2. Последний результат станет baseline
tail -1 benchmarks/results_history.jsonl | python -m json.tool

# 3. Сохраните как baseline
python -c "
import json
with open('benchmarks/results_history.jsonl', 'r') as f:
    latest = json.loads(f.readlines()[-1])

baseline = {
    'date': latest['timestamp'],
    'version': 'v1.0.0',
    'dataset': 'golden-41-questions',
    'dataset_size': latest['dataset_size'],
    'metrics': latest['aggregate_metrics']
}

with open('benchmarks/baseline.json', 'w') as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print('✅ Baseline создан!')
print(f\"Faithfulness: {baseline['metrics'].get('mean_faithfulness_score', 0):.2%}\")
print(f\"Citation Rate: {baseline['metrics'].get('citation_rate', 0):.2%}\")
"

# 4. Теперь используйте эти цифры в резюме!
```

---

## 🐛 Troubleshooting

### Проблема: ModuleNotFoundError

```bash
# Переустановите зависимости
pip install -r requirements.txt --upgrade
```

### Проблема: API ключи не работают

```bash
# Проверьте .env
cat .env | grep -E "API_KEY|CREDENTIALS"

# Убедитесь что ключи без кавычек
# ✅ Правильно: OPENAI_API_KEY=sk-abc123
# ❌ Неправильно: OPENAI_API_KEY="sk-abc123"
```

### Проблема: ChromaDB не найдена

```bash
# Запустите индексацию
python index.py

# Проверьте что создалась БД
ls -lh chroma_db_*/
```

### Проблема: Тесты падают

```bash
# Проверьте версию Python (нужна 3.11+)
python --version

# Переустановите pytest
pip install pytest pytest-cov --upgrade

# Запустите с verbose
pytest tests/ -v --tb=short
```

---

## 📦 Структура после установки

```
safety-incident-analyzer/
├── venv/                          # Виртуальное окружение
├── .env                           # API ключи (создается вручную)
├── source_docs/                   # Ваши документы
├── chroma_db_*/                   # Векторная БД (создается index.py)
├── benchmarks/
│   ├── baseline.json              # Baseline метрики
│   └── results_history.jsonl      # История оценок
├── tests/
│   ├── dataset.csv                # 41 золотой вопрос
│   └── test_*.py                  # Unit тесты
└── htmlcov/                       # Отчет покрытия тестами
```

---

## ✅ Готово!

После выполнения всех шагов у вас будет:

- ✅ Локальная копия проекта
- ✅ Рабочее окружение с зависимостями
- ✅ Все тесты проходят
- ✅ Baseline метрики для резюме
- ✅ Работающее Streamlit приложение

**Для резюме возьмите цифры из:**
```bash
cat benchmarks/baseline.json | grep -E "faithfulness|citation"
```

И используйте в описании проекта! 🎉
