# Как работает V7 Pipeline

> Детальное объяснение логики — для интервью, ревью, онбординга.

---

## Используемые техники

### Retrieval
- **Гибридный поиск** — векторный (ChromaDB, cosine similarity 0–1) + BM25 (ключевые слова, бесшкальный)
- **RRF (Reciprocal Rank Fusion)** — слияние двух списков без настройки весов: `score = Σ 1/(rank + 60)`
- **FlashRank Reranking** — cross-encoder переранжирование топ-результатов после hybrid merge
- **Query Expansion** — дополнительные термины из найденных документов (в RAG Complex)
- **Multi-attempt merge** — слияние результатов нескольких попыток поиска, top_k=24

### Качество ответов
- **Hard Gates** — детерминированные пороги (score, кол-во чанков, keyword overlap) без LLM-решений
- **3-way Triage** — маршрутизация по уверенности: sufficient / borderline / clearly_bad
- **Document Diversity** — защита от ответа только из одного источника (max_doc_ratio ≤ 0.7)
- **Abstain** — явный отказ при недостатке данных вместо галлюцинации

### NLP и безопасность
- **Term Glossary** — stem-based расшифровка аббревиатур для русского языка перед поиском
- **Prompt Injection фильтр** — `sanitize_for_llm()` удаляет "ignore previous instructions" и подобное
- **NoSQL Injection whitelist** — валидация фильтров ChromaDB через `ALLOWED_FILTER_KEYS`

### Chunking и индексация
- **Docling** — парсинг PDF/DOCX с сохранением структуры документа
- **Chunk 1200 / overlap 150** — параметры подобраны под нормативные тексты (длинные абзацы с определениями)

### LLM и оркестрация
- **Gemini Flash, thinking_budget=4096** — управляемая глубина рассуждений модели
- **Dependency Injection (bridge.py)** — LLM и vector search инжектятся в граф, не захардкожены → легко менять провайдера
- **LangGraph StateGraph** — детерминированный граф состояний
- **LangSmith** — трассировка каждого запроса в production

### Eval
- **166 unit-тестов** — покрытие nlp_core, hard_gates, всех нод графа
- **A/B тестирование** — `run_ab_test.py` + LangSmith для сравнения версий

---

## Зачем V7?

Проблема предыдущих версий: система всегда отвечала, даже если ничего не нашла.
Нужна была детерминированная логика: "нашли достаточно → отвечаем, не нашли → честно говорим что не знаем".

V7 решает это через **hard gates** — числовые пороги без LLM-решений. Граф детерминирован: при одинаковом запросе и базе путь всегда одинаковый.

---

## Схема пути запроса

```
Запрос
  → Глоссарий терминов       (детерминированный: "программа А" → официальный термин)
  → Intent Gate              (regex: шум/приветствие → Отказ, нормативный → дальше)
  → RAG Simple               (быстрый гибридный поиск)
      → Evaluate Triage      (Гейт #1)
          ДА (score ≥ 0.50)  → Generate Answer  ← быстрый путь
          НЕТ (borderline)   → RAG Complex
  → RAG Complex              (глубокий поиск + расширение запроса)
      → Evaluate Complex     (Гейт #2)
          ПРОЙДЕН            → Generate Answer
          ПРОВАЛЕН           → Abstain  ("не нашли достаточно данных")
  → Generate Answer          (Gemini Flash, thinking_budget=4096)
  → Финальный ответ + источники
```

---

## Hard Gates — что это и как работают

Hard gate — это функция `check_hard_gates()` в `src/v7/hard_gates.py`.

Она принимает список найденных фрагментов (passages) и план (пороги) и проверяет **три условия одновременно**. Все три должны быть True — иначе `sufficient=False`.

### Три условия

| Условие | Что проверяет | Порог (rag_simple) | Порог (rag_complex) |
|---|---|---|---|
| `above_threshold` | top_score ≥ threshold | **0.50** | **0.35** |
| `enough_evidence` | кол-во чанков ≥ min_passages | 1 (simple path) | **8** |
| `keyword_overlap_ok` | доля ключевых слов запроса, найденных в чанках | 0.0 (отключён) | **0.20** |

```python
# Из hard_gates.py — упрощённо:
sufficient = all([
    top_score >= plan["threshold"],          # score от ChromaDB (0-1)
    len(passages) >= plan["min_passages"],    # количество чанков
    keyword_overlap >= plan["min_kw_overlap"] # доля слов запроса в тексте
])
```

### Откуда берётся score?

Score — это **cosine similarity** из ChromaDB (диапазон 0–1). **Важно:** BM25 scores не используются для порогов — они бесшкальные (0–20+). В `rag_simple.py` `top_score` берётся только из `vector_results`, не из merged.

### Почему два разных порога?

`rag_simple` — быстрый путь. Порог 0.50 — высокий, значит нашли что-то явно релевантное.

`rag_complex` — fallback. Порог 0.35 — ниже, но добавляются требования к количеству чанков (8+) и keyword overlap (20%+). Смысл: если score чуть ниже, компенсируем объёмом и лексическим покрытием.

---

## Triage — три категории после RAG Simple

После `rag_simple` нода `evaluate_triage` классифицирует результат:

| Категория | Условие | Что происходит |
|---|---|---|
| `sufficient` | hard gates OK | → Generate Answer напрямую |
| `borderline` | score в зоне 0.38–0.50 | → RAG Complex (попробуем лучше) |
| `clearly_bad` | score < 0.38 или мало чанков | → RAG Complex |

Зона borderline (0.38–0.50) — "может найдём лучше". Ниже 0.38 — явно плохо, но всё равно идём в Complex, там другие параметры поиска.

---

## RAG Complex — что делает иначе

`src/v7/nodes/rag_complex.py` запускает поиск по-другому:

1. **Расширение запроса через BM25** — дополнительные термины из топ-документов
2. **Несколько попыток** — с разными параметрами (разный top_k, разные фильтры)
3. **Merge всех попыток** — `merge_all_passages(attempts, top_k=24)` в `evaluate_complex`

`top_k=24` — важный параметр. Раньше было 12 — ответы были неполными (система находила 3 из 8 категорий). После увеличения до 24 — все категории в ответе.

---

## Дополнительные защиты

### Prompt Injection (в `hard_gates.py`)
```python
# Паттерны фильтруются до передачи в Gemini:
"ignore previous instructions" → "[FILTERED]"
"system:" → "[FILTERED]"
"you are now" → "[FILTERED]"
```

### NoSQL Injection (validate_filters)
Фильтры для ChromaDB проходят whitelist-валидацию — только разрешённые ключи (`ALLOWED_FILTER_KEYS`). Произвольные where-clause не проходят.

### Document Diversity
Если все 8+ чанков из одного документа — это `max_doc_ratio > 0.7`, escalation_hint = True. При multi-doc запросах это делает hard gate провальным (diversity — hard требование, а не совет).

---

## Generate Answer — что внутри

`src/v7/nodes/generate_answer.py` вызывает `make_generate_fn()` из `bridge.py`.

Bridge инжектит Gemini Flash с `thinking_budget=4096`. Модель получает:
- Финальные passages (от 1 до 24 чанков)
- Запрос пользователя
- Промпт с инструкцией "отвечай только по документам, цитируй источники"

**Fallback:** если Gemini вернул 503 (перегружен) — возвращается stub (сырые тексты чанков без синтеза). **P0 в backlog:** добавить retry с tenacity.

---

## Конфигурация порогов

Все пороги в `src/v7/config.py`, переопределяются через env:

```env
V7_HARD_GATE_THRESHOLD=0.50       # порог для rag_simple
V7_TRIAGE_SOFT_THRESHOLD=0.38     # нижняя граница borderline
V7_COMPLEX_THRESHOLD=0.35         # порог для rag_complex
V7_COMPLEX_MIN_PASSAGES=8         # мин. чанков для rag_complex
V7_COMPLEX_MIN_KW_OVERLAP=0.20    # мин. keyword overlap для rag_complex
```

Smoke test для проверки пайплайна целиком:
```bash
python scripts/trace_v7.py "кто должен обучаться по программе А охраны труда?"
```
