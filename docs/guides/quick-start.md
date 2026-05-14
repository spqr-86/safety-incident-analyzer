# 🚀 Quick Start

Установка и первый запуск проекта локально.

## Требования

- Python 3.11+
- API-ключи: OpenAI (эмбеддинги + опционально LLM) и Google Gemini (генерация в V7)

## 1. Установка

```bash
git clone https://github.com/spqr-86/safety-incident-analyzer.git
cd safety-incident-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Настройка окружения

Скопируйте `.env.example` → `.env` и заполните ключи:

```env
# LLM / эмбеддинги
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
EMBEDDING_PROVIDER=openai          # openai | hf_api | local

# Gemini — генерация и LLM-судьи в V7
GEMINI_API_KEY=your_gemini_key
GEMINI_FAST_MODEL=gemini-3-flash-preview

# Включить V7-граф в UI
USE_V7_GRAPH=true
```

Ключевые настройки — в `config/settings.py` (общие) и `src/v7/config.py` (пороги V7,
env-префикс `V7_`).

## 3. Индексация документов

Положите нормативные документы (PDF / DOCX / MD) в `source_docs/` и запустите:

```bash
python index.py
```

> ⚠️ `index.py` **destructive** — дропает текущую коллекцию ChromaDB перед переиндексацией.

Пайплайн индексации: Docling → препроцессинг → chunking (1500 / 400) → OpenAI
embeddings → ChromaDB. Подробно — [DATA_PIPELINE.md](../DATA_PIPELINE.md).

## 4. Запуск приложения

```bash
streamlit run app.py --server.port 8502
```

Откройте `http://localhost:8502`, задайте вопрос в чате — система пройдёт V7-граф
(`intent_gate → router → rag_simple → … → generate_answer`) и вернёт ответ со
ссылками на источники.

## 5. Прогон eval (опционально)

```bash
python eval/run_v7_eval.py                 # весь golden-датасет
python eval/run_v7_eval.py --limit 5       # быстрый smoke-тест
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

Метрики и формат отчёта — [docs/evaluation/README.md](../evaluation/README.md).

## 6. Тесты

```bash
pytest                       # все
pytest -m unit               # только unit
python scripts/trace_v7.py "для кого проводится повторный инструктаж?"   # E2E-трассировка
```

## Куда дальше

- [Архитектура](../architecture/README.md) · [Как работает V7](../architecture/v7-how-it-works.md)
- [Eval framework](../evaluation/README.md)
- [Добавление вопросов в датасет](./adding-questions.md)
