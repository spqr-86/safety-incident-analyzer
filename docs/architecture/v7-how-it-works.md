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
- **Term Glossary** — расшифровка доменных аббревиатур (`src/glossary.py`); применяется в ноде `router`. Слова >4 букв матчатся по морфологическому стему, аббревиатуры ≤4 букв — целым словом
- **Prompt Injection фильтр** — `sanitize_for_llm()` удаляет "ignore previous instructions" и подобное
- **NoSQL Injection whitelist** — валидация фильтров ChromaDB через `ALLOWED_FILTER_KEYS`

### Chunking и индексация
- **Docling** — парсинг PDF/DOCX с сохранением структуры документа
- **Chunk 1500 / overlap 400** (`config/settings.py`) — параметры подобраны под нормативные тексты (длинные абзацы с определениями)

### LLM и оркестрация
- **Gemini (thinking_budget=4096)** — управляемая глубина рассуждений; модель из `GEMINI_FAST_MODEL`
- **Dependency Injection (bridge.py)** — LLM и vector search инжектятся в граф, не захардкожены → легко менять провайдера
- **LangGraph StateGraph** — детерминированный граф состояний

### Eval
- **`eval/run_v7_eval.py`** — прогон golden-датасета через V7-граф, LLM-as-judge метрики. См. [docs/evaluation/README.md](../evaluation/README.md)

---

## Зачем V7?

Проблема предыдущих версий: система всегда отвечала, даже если ничего не нашла.
Нужна была детерминированная логика: "нашли достаточно → отвечаем, не нашли → честно говорим что не знаем".

V7 решает это через **hard gates** — числовые пороги без LLM-решений. Граф детерминирован: при одинаковом запросе и базе путь всегда одинаковый.

---

## Схема пути запроса

```
Запрос
  → Intent Gate              (regex: шум → END, нормативный → Router)
  → Router                   (классификация запроса, plan, глоссарий → active_query)
  → RAG Simple               (быстрый гибридный поиск, SIMPLE_TOP_K=12)
      → Evaluate Triage      (Гейт #1)
          sufficient (score ≥ 0.50)   → Generate Answer  ← быстрый путь
          borderline (0.38–0.50)      → LLM Verifier → (ok → Generate / rewrite → Rewriter → RAG Simple / escalate → RAG Complex)
          clearly_bad (< 0.38)        → RAG Complex
  → RAG Complex              (глубокий поиск COMPLEX_TOP_K=60 + rerank + MMR)
      → Evaluate Complex     (Гейт #2)
          ПРОЙДЕН            → Generate Answer
          ПРОВАЛЕН           → Abstain  ("не нашли достаточно данных")
  → Visual Enrichment        (опционально, перед генерацией)
  → Generate Answer          (Gemini, thinking_budget=4096)
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
| `enough_evidence` | кол-во чанков ≥ min_passages | **5** | **8** |
| `keyword_overlap_ok` | доля ключевых слов запроса, найденных в чанках | **0.15** | **0.20** |

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
| `borderline` | score в зоне 0.38–0.50 | → LLM Verifier (решает: ответить / переформулировать / эскалировать) |
| `clearly_bad` | score < 0.38 или мало чанков | → RAG Complex |

Зона borderline (0.38–0.50) — "может найдём лучше": `llm_verifier` смотрит passages и
решает — ответ годен (→ generate), нужна переформулировка (→ `rewriter` → RAG Simple)
или эскалация (→ RAG Complex). Ниже 0.38 — явно плохо, сразу в Complex.

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

## Visual Enrichment — визуальные доказательства

Нода `src/v7/nodes/visual_enrichment.py` вставлена **перед `generate_answer`** — после evaluate_triage/verifier/evaluate_complex.

Цель: добавить к текстовым чанкам визуальный контекст (скриншоты таблиц, страниц), чтобы Gemini мог ответить точнее на вопросы где важна структура документа.

### Когда срабатывает

Нода анализирует каждый passage из найденных чанков и принимает решение по триггерам:

| Триггер | Условие | Действие |
|---|---|---|
| Таблица | `element_type == "Table"` | `mode=analyze` — VLM анализирует таблицу и добавляет текстовое описание |
| Короткий чанк | `len(text) < 150` | `mode=show` — передаёт image_path, пусть модель видит оригинал |
| Неполный текст | `detect_incomplete_chunk()` — обрыв на ":" или "№" | `mode=show` — текст явно обрезан, нужен оригинал |

### Ограничения

- **MAX_VISUAL_PROOFS = 3** — не более 3 визуальных доказательств на запрос (экономия токенов)
- Нода **no-op** если не инжектирована `visual_proof_fn` — безопасно в окружениях без VLM
- Исключения на отдельном passage не останавливают остальные

### Dependency Injection

```python
# bridge.py — init_v7_from_chroma()
visual_proof_fn = make_visual_proof_fn()          # без аргументов; None если agent_tools недоступен
if visual_proof_fn is not None:
    visual_enrichment_mod.set_visual_proof_fn(visual_proof_fn)
```

На VPS нет `agent_tools` (VLM не настроен) → `make_visual_proof_fn()` возвращает None,
нода пропускается без ошибок.

### Пример в трассировке

```
[visual_enrichment] passage 0: element_type=Table → analyze → добавлен visual_context
[visual_enrichment] passage 2: len=87 → show → image_path передан
[visual_enrichment] passage 5: incomplete chunk → show → image_path передан
```

---

## Generate Answer — что внутри

`src/v7/nodes/generate_answer.py` вызывает `make_generate_fn()` из `bridge.py`.

Bridge инжектит Gemini с `thinking_budget=4096`. Модель получает:
- Финальные passages (до 24 чанков; `make_generate_fn` берёт `final_passages[:24]`)
- Запрос пользователя
- Промпт `_GENERATE_SYSTEM_PROMPT` (хардкод-строка в `bridge.py`): "отвечай только по документам, цитируй источники"

**Retry:** если Gemini вернул 503 — `tenacity` делает 3 попытки с экспоненциальным backoff (2→4→8s). Только после всех ретраев возвращается stub (сырые тексты чанков без синтеза).

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
