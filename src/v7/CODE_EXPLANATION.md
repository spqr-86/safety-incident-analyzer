# Подробное объяснение кода V7 RAG Pipeline

## Содержание

1. [Общий обзор проекта](#1-общий-обзор-проекта)
2. [Архитектура V7 — зачем и почему](#2-архитектура-v7--зачем-и-почему)
3. [Граф зависимостей модулей](#3-граф-зависимостей-модулей)
4. [Типы данных и состояние графа (state_types.py)](#4-типы-данных-и-состояние-графа)
5. [Конфигурация (config.py)](#5-конфигурация)
6. [NLP-ядро (nlp_core.py)](#6-nlp-ядро)
7. [Hard Gates и Triage (hard_gates.py)](#7-hard-gates-и-triage)
8. [Ноды графа (nodes/)](#8-ноды-графа)
9. [Сборка графа (graph.py)](#9-сборка-графа)
10. [Мост к инфраструктуре (bridge.py)](#10-мост-к-инфраструктуре)
11. [Полный путь запроса — пошагово](#11-полный-путь-запроса--пошагово)
12. [Внешние зависимости](#12-внешние-зависимости)
13. [Тестирование](#13-тестирование)
14. [Как читать и модифицировать код](#14-как-читать-и-модифицировать-код)

---

## 1. Общий обзор проекта

**AI Safety Compliance Assistant** — система ответов на вопросы по российским нормативным документам (ГОСТ, СНиП, СП, ФЗ и др.). Пользователь задаёт вопрос на русском языке, система ищет релевантные фрагменты в базе документов и формирует ответ.

### Стек технологий

| Компонент | Технология |
|-----------|------------|
| Оркестрация агентов | LangGraph (StateGraph) |
| Векторная БД | ChromaDB |
| LLM | Google Gemini (основной), OpenAI (fallback) |
| Эмбеддинги | multilingual-e5-base / OpenAI |
| Полнотекстовый поиск | BM25 через rank_bm25 |
| Морфология русского | pymorphy3 + razdel |
| UI | Streamlit |
| Конфигурация | pydantic-settings |
| Промпты | Jinja2 шаблоны с реестром версий |

### Три режима работы

1. **V7 Graph** (`src/v7/`) — новая модульная архитектура (этот документ).
2. **Multi-Agent RAG** (`agents/multiagent_rag.py`) — ReAct-агент с верификатором.
3. **Simple RAG Chain** (`src/final_chain.py`) — legacy fallback (linear chain).

Переключение между режимами: настройка `USE_V7_GRAPH` в `.env`.

---

## 2. Архитектура V7 — зачем и почему

### Проблемы предыдущей версии (Multi-Agent RAG)

Предыдущая архитектура (`agents/multiagent_rag.py`, ~37KB в одном файле) имела проблемы:
- Монолитный код — сложно тестировать и расширять
- ReAct-агент непредсказуем — может зациклиться
- Нет явных gates — система не знает, когда results "достаточно хороши"
- Жёсткая связь с LLM — нельзя протестировать без API-ключей

### Принципы V7

1. **Строгий граф зависимостей**: `state_types ← config ← nlp_core ← hard_gates ← nodes ← graph`. Обратных импортов нет.
2. **Тонкие ноды**: каждая нода только оркестрирует (read state → call function → write state). Вся бизнес-логика в `nlp_core` и `hard_gates`.
3. **Dependency Injection**: внешние зависимости (vector search, LLM verify, LLM rewrite) инжектируются через `set_*()` функции. Для тестов используются стабы.
4. **Не изобретать велосипед**: `rank_bm25` для BM25, `pymorphy3` для морфологии, `razdel` для токенизации, `pydantic` для конфигурации.

---

## 3. Граф зависимостей модулей

```
state_types.py          ← Нулевой уровень. Никаких импортов из проекта.
     ↑
config.py               ← Только pydantic-settings.
     ↑
nlp_core.py             ← Импортирует config. Морфология, BM25, RRF, MMR.
     ↑
hard_gates.py           ← Импортирует nlp_core + state_types. Gates и triage.
     ↑
nodes/*.py              ← Импортируют hard_gates, nlp_core, state_types.
     ↑
graph.py                ← Импортирует все nodes. Собирает StateGraph.
     ↑
bridge.py               ← Импортирует nodes, nlp_core, graph. Связь с ChromaDB/LLM.
     ↑
__init__.py             ← Публичный API пакета.
```

**Правило**: стрелка всегда направлена вверх. Нижний модуль может импортировать верхний, но не наоборот.

---

## 4. Типы данных и состояние графа

**Файл**: `src/v7/state_types.py` (~180 строк)

Это фундамент всей системы. Определяет все структуры данных, используемые в графе.

### Data-классы для документов

```python
@dataclass
class Doc:
    id: str          # уникальный ID документа
    text: str        # текст фрагмента
    metadata: dict   # метаданные (doc_type, section, year и т.д.)

@dataclass
class ScoredDoc(Doc):
    score: float     # оценка релевантности (0.0–1.0)
```

### Literal-типы (ограничения на значения)

```python
Intent = Literal["noise", "domain"]                          # шум или доменный запрос
TriageCategory = Literal["sufficient", "borderline", "clearly_bad"]  # 3-way triage
VerifierVerdict = Literal["sufficient", "rewrite", "escalate"]       # вердикт LLM
```

Каждый переход в графе типизирован:
```python
NextAfterIntent   = Literal["end", "router"]
NextAfterRouter   = Literal["rag_simple", "clarify_respond"]
NextAfterTriage   = Literal["end", "llm_verifier", "rag_complex"]
NextAfterVerifier = Literal["end", "rewriter", "rag_complex"]
NextAfterEvalComplex = Literal["end", "abstain"]
```

Это делает переходы между нодами **детерминированными** и **верифицируемыми** — нельзя случайно отправить данные по несуществующему маршруту.

### RetrievalPlan — план поиска

```python
class RetrievalPlan(TypedDict, total=False):
    top_k: int                    # сколько фрагментов запросить
    rerank: bool                  # использовать ли reranking
    timeout_ms: int               # таймаут retrieval
    threshold: float              # минимальный score для hard gate
    min_passages: int             # минимальное количество фрагментов
    min_keyword_overlap: float    # минимальное совпадение ключевых слов
    max_single_doc_ratio: float   # максимальная доля одного документа (diversity)
    borderline_threshold: float   # порог для "clearly_bad" зоны
    min_verifier_confidence: float # минимальная уверенность LLM-верификатора
    require_multi_doc: bool       # для запросов-сравнений
    mmr_lambda: float             # параметр MMR (1.0 = только релевантность)
```

Plan создаётся нодой `router` и обновляется при эскалации в `rag_complex`.

### RetrievalAttempt — результат одной попытки

```python
class RetrievalAttempt(TypedDict, total=False):
    retrieval_id: str       # SHA256-хеш запроса + фильтров
    stage: Literal["simple", "complex"]
    passages: List[dict]    # найденные фрагменты
    top_score: float        # лучший score
    attempt_plan: dict      # snapshot плана на момент retrieval
    metrics: dict           # метрики для offline evaluation
```

Важно: `retrieval_attempts` — **append-only** список. Каждая попытка добавляется, а не заменяет предыдущую. Это позволяет `evaluate_complex` объединять passages из всех попыток.

### HardGateResult и SufficiencyResult

```python
class HardGateResult(TypedDict):
    sufficient: bool              # все hard gates пройдены?
    above_threshold: bool         # top_score >= threshold?
    enough_evidence: bool         # passage_count >= min_passages?
    keyword_overlap_ok: bool      # keyword overlap >= min_keyword_overlap?
    top_score: float
    passage_count: int
    keyword_overlap_active: float   # overlap по active_query
    keyword_overlap_original: float # overlap по original query (drift detection)
```

`SufficiencyResult` расширяет `HardGateResult` полями diversity и triage:
- `diversity_ok`, `escalation_hint` — soft signals
- `triage` — итоговая 3-way категория
- `unique_docs`, `max_doc_ratio` — метрики diversity

### RAGState — полное состояние графа

```python
class RAGState(TypedDict, total=False):
    # INPUT (задаётся вызывающим кодом, immutable)
    query: str                      # исходный запрос пользователя
    filters: dict                   # фильтры (doc_type, year и т.д.)

    # INTERNAL (модифицируется нодами)
    intent: Intent                  # noise / domain
    plan: RetrievalPlan             # текущий план поиска
    retrieval_id: str               # SHA256 хеш для дедупликации
    active_query: str               # рабочий запрос (может быть переформулирован)
    retrieval_attempts: Annotated[List[RetrievalAttempt], operator.add]  # append-only!
    sufficient: bool                # пройдены ли все проверки?
    verify_iteration: int           # счётчик rewrite-итераций
    verification: VerificationResult # результат LLM-верификации

    # OUTPUT (финальные результаты)
    final_passages: List[dict]      # финальные фрагменты для ответа
    final_score: float              # лучший score
    fallback_passages: List[dict]   # запасные фрагменты (из fast-path)
    fallback_score: float
    clarify_message: str            # запрос уточнения
    abstain_reason: str             # причина отказа
    sufficiency_details: SufficiencyResult

    # UX
    status_message: str             # прогресс для frontend
```

Обратите внимание на `Annotated[List[RetrievalAttempt], operator.add]` — это указание LangGraph, что поле нужно **объединять** (конкатенировать списки), а не заменять. Каждая нода может вернуть `{"retrieval_attempts": [new_attempt]}`, и LangGraph автоматически добавит его к существующему списку.

---

## 5. Конфигурация

**Файл**: `src/v7/config.py` (~50 строк)

Все пороги вынесены в `V7Config` через `pydantic-settings`:

```python
class V7Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="V7_", ...)

    HARD_GATE_THRESHOLD: float = 0.65
    TRIAGE_SOFT_THRESHOLD: float = 0.40
    RRF_K: int = 60               # k для Reciprocal Rank Fusion
    MMR_LAMBDA: float = 0.7       # баланс relevance vs diversity
    BM25_TOP_K: int = 20
    SEMANTIC_TOP_K: int = 20
    MIN_KEYWORD_OVERLAP_ACTIVE: float = 0.3
    MIN_KEYWORD_OVERLAP_ORIGINAL: float = 0.2
    MAX_REWRITE_ATTEMPTS: int = 2
    VERIFIER_CONFIDENCE_ANCHOR: float = 0.7
    MAX_INPUT_LENGTH: int = 2000
    BLOCKED_PATTERNS: list[str] = [...]  # anti-injection
```

Любой параметр можно переопределить через переменную окружения с префиксом `V7_`:
```bash
V7_HARD_GATE_THRESHOLD=0.80  # повысить порог для hard gate
```

Синглтон `v7_config = V7Config()` создаётся при импорте модуля.

---

## 6. NLP-ядро

**Файл**: `src/v7/nlp_core.py` (~335 строк)

Содержит всю NLP-логику: морфология, BM25, RRF, MMR. Ни одна нода не содержит NLP-кода — всё делегируется сюда.

### 6.1. Морфологический анализатор

```python
_morph = pymorphy3.MorphAnalyzer()  # singleton, тяжёлый объект
```

Используется для лемматизации русских слов. Например: "требований" → "требование", "противопожарной" → "противопожарный".

### 6.2. extract_keywords(text) → set[str]

Извлекает ключевые слова из текста:
1. Извлекает номера документов regex-ом (например, "1.13130" из "СП 1.13130")
2. Токенизирует текст через `razdel`
3. Лемматизирует каждый токен через `pymorphy3`
4. Фильтрует стоп-слова (frozenset из ~40 русских слов)
5. Возвращает объединение лемм и номеров документов

### 6.3. compute_keyword_overlap(query, passages) → float

Считает долю ключевых слов запроса, найденных в passages:
```
overlap = |keywords(query) ∩ keywords(passages)| / |keywords(query)|
```
Возвращает значение от 0.0 до 1.0. Используется как hard gate.

### 6.4. compute_doc_diversity(passages) → (unique_docs, max_ratio)

Считает разнообразие источников:
- `unique_docs` — количество уникальных документов
- `max_ratio` — доля самого частого документа (1.0 = все из одного документа)

Используется как soft signal в triage.

### 6.5. BM25Index

Обёртка над `rank_bm25.BM25Okapi` с русской лемматизацией:

```python
class BM25Index:
    def __init__(self, passages: List[dict]):
        # Лемматизируем корпус при инициализации
        corpus = [_lemmatize_for_bm25(p["text"]) for p in passages]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=12, filters=None) -> List[dict]:
        # Лемматизируем запрос → BM25 scoring → фильтрация → top_k
```

Глобальный индекс инициализируется один раз при старте через `init_bm25_index()`.

### 6.6. rrf_merge(*result_lists, top_k, k) → List[dict]

**Reciprocal Rank Fusion** — объединяет результаты из нескольких retrievers:

```
RRF_score(doc) = Σ_i 1/(k + rank_i(doc))
```

Где `k=60` (стандартное значение из литературы). Дедупликация по `chunk_id`.

Зачем: vector search хорошо находит семантически близкие тексты, BM25 — тексты с точными терминами. RRF объединяет сильные стороны обоих.

### 6.7. mmr_select(passages, top_k, lambda_param) → List[dict]

**Maximal Marginal Relevance** — выбирает passages, балансируя релевантность и разнообразие:

```
MMR_score = λ * relevance - (1-λ) * diversity_penalty
```

Diversity penalty рассчитывается на основе `doc_id` — штрафуем фрагменты из документа, который уже хорошо представлен. Используется как fallback в `merge_all_passages()`.

### 6.8. merge_all_passages(attempts, top_k, mmr_lambda) → List[dict]

Финальное объединение passages из всех попыток retrieval:
1. Собрать passages из всех RetrievalAttempt
2. Дедуплицировать по `chunk_id`
3. Применить MMR для diversity
4. Вернуть top_k

---

## 7. Hard Gates и Triage

**Файл**: `src/v7/hard_gates.py` (~220 строк)

### 7.1. validate_filters(filters) → dict | None

Whitelist-валидация фильтров. Допустимые ключи: `doc_type`, `doc_id`, `section`, `category`, `year`. Всё остальное отбрасывается — защита от NoSQL injection через ChromaDB where-clause.

### 7.2. sanitize_for_llm(text) → str

Удаляет prompt injection паттерны из текста перед отправкой в LLM:
- "ignore previous instructions"
- "system:"
- "you are now"
- "new instructions:"
- "forget everything"

### 7.3. check_hard_gates(...) → HardGateResult

Проверяет три обязательных условия (**все** должны быть True):

| Gate | Условие | Что проверяет |
|------|---------|---------------|
| `above_threshold` | `top_score >= plan.threshold` | Есть ли хотя бы один релевантный passage |
| `enough_evidence` | `len(passages) >= plan.min_passages` | Достаточно ли фрагментов |
| `keyword_overlap_ok` | `overlap >= plan.min_keyword_overlap` | Содержат ли passages ключевые слова запроса |

Дополнительно вычисляется **dual keyword overlap**:
- `keyword_overlap_active` — по текущему (возможно переформулированному) запросу → для hard gate
- `keyword_overlap_original` — по исходному запросу → для drift detection

### 7.4. check_full_triage(...) → SufficiencyResult

Расширяет hard gates soft signals и 3-way triage:

```
                          ┌──────────────────────┐
                          │   Hard Gates OK?     │
                          └──────┬───────────────┘
                                 │
                  ┌──────────────┼───────────────┐
                  │ yes          │ partial        │ no
                  │              │                │
         ┌────────▼──────┐ ┌────▼──────┐ ┌──────▼──────────┐
         │ Diversity OK? │ │ Score в   │ │ Score <          │
         │               │ │ borderline│ │ borderline       │
         └───┬─────┬─────┘ │ зоне?    │ │ ИЛИ мало passages│
             │     │       └────┬──────┘ └──────┬───────────┘
          yes│     │no          │               │
             │     │            │               │
      ┌──────▼──┐ ┌▼──────────▼┐        ┌─────▼──────┐
      │sufficient│ │ borderline  │        │ clearly_bad │
      └─────────┘ └─────────────┘        └────────────┘
```

- **sufficient**: всё хорошо, можно формировать ответ
- **borderline**: есть что-то полезное, но стоит перепроверить через LLM
- **clearly_bad**: ничего полезного не найдено, нужна эскалация

### 7.5. compute_attempt_metrics(...) → (HardGateResult, dict)

Утилита для вычисления hard gates + метрик для RetrievalAttempt. Переиспользуется в `rag_simple`, `rag_complex` и `evaluate_*`.

---

## 8. Ноды графа

**Директория**: `src/v7/nodes/`

Все ноды следуют паттерну **thin node**:
```python
def node_name(state: RAGState) -> RAGState:
    # 1. Read state
    data = state["field"]
    # 2. Call nlp_core / hard_gates
    result = some_function(data)
    # 3. Write state
    return {"output_field": result}
```

### 8.1. intent_gate — фильтр шума

**Файл**: `nodes/intent_gate.py` (20 строк)

Первая нода в графе. Классифицирует запрос как `noise` или `domain`:
- Запросы короче 3 символов → `noise`
- "привет", "hello", "как дела" и т.д. → `noise`
- Всё остальное → `domain`

```
intent_gate → "noise" → END (ничего не делаем)
intent_gate → "domain" → router (продолжаем)
```

### 8.2. router — классификация и планирование

**Файл**: `nodes/router.py` (105 строк)

Определяет тип запроса и создаёт RetrievalPlan:

**Классификация** (keyword-based):
- Маркеры сравнения ("сравни", "разница", "отличие", "vs") → `require_multi_doc=True`, `mmr_lambda=0.5`
- Маркеры фактоида ("пункт", "таблица", "значение", "не менее") → `mmr_lambda=0.95`
- По умолчанию → `mmr_lambda=0.7`

**Short query guard**: запросы короче 8 символов → `clarify_message` → END.

**Создаёт plan** с дефолтными порогами для fast-path:
```python
plan = {
    "top_k": 12,           # запросить 12 passages
    "threshold": 0.45,     # минимальный score
    "min_passages": 5,     # минимум 5 фрагментов
    "min_keyword_overlap": 0.3,  # 30% overlap
    ...
}
```

### 8.3. rag_simple — быстрый гибридный поиск

**Файл**: `nodes/rag_simple.py` (84 строки)

Выполняет **hybrid retrieval**: vector search + BM25 → RRF merge.

Последовательность:
1. Проверка дедупликации (не повторять тот же `retrieval_id` + `stage`)
2. Vector search через инжектированную функцию `_vector_search()`
3. BM25 full-text search через `bm25_search()`
4. RRF merge результатов
5. Если RRF пустой — fallback на vector results
6. Вычисление метрик через `compute_attempt_metrics()`
7. Запись RetrievalAttempt в состояние

**Dependency Injection**: функция `_vector_search` по умолчанию возвращает `[]`. Реальная реализация инжектируется через `set_vector_search()` при инициализации.

### 8.4. evaluate_triage — трёхсторонний gate

**Файл**: `nodes/evaluate_triage.py` (58 строк)

Оценивает качество результатов `rag_simple`:

```
evaluate_triage
    ├─ "sufficient"    → записать final_passages → END
    ├─ "borderline"    → llm_verifier (перепроверить через LLM)
    └─ "clearly_bad"   → rag_complex (тяжёлый поиск)
```

Если hard gates прошли, но triage ≠ sufficient (из-за soft signals) — сохраняет `fallback_passages` на случай, если дальнейшие этапы тоже провалятся.

### 8.5. llm_verifier — LLM-верификация

**Файл**: `nodes/llm_verifier.py` (153 строки)

LLM оценивает, достаточно ли passages для ответа на запрос. Ожидает JSON:

```json
{
  "verdict": "sufficient | rewrite | escalate",
  "reason": "обоснование",
  "rewrite_hint": "что искать иначе",
  "missing_aspects": ["список недостающих аспектов"],
  "confidence": 0.0-1.0
}
```

**Три protection gate:**

1. **Error gate**: если LLM недоступен → автоматический `escalate`
2. **Confidence gate**: если `confidence < min_verifier_confidence` → принудительный `escalate`
3. **Max iterations gate**: если `verdict == "rewrite"` и `iteration >= MAX_VERIFY_ITERATIONS (2)` → принудительный `escalate`

```
llm_verifier
    ├─ "sufficient"  → записать final_passages → END
    ├─ "rewrite"     → rewriter (переформулировать запрос)
    └─ "escalate"    → rag_complex (тяжёлый поиск)
```

### 8.6. rewriter — переформулировка запроса

**Файл**: `nodes/rewriter.py` (63 строки)

Переформулирует запрос на основе `rewrite_hint` и `missing_aspects` от верификатора.

**Защита от query drift**: все идентификаторы документов (ГОСТ, СП, СНиП и т.д.) извлекаются из оригинального запроса и гарантированно сохраняются в переформулированном.

После rewrite:
- `active_query` обновляется
- `retrieval_id` пересчитывается
- `verify_iteration` инкрементируется
- Граф возвращается к `rag_simple` (новый поиск с новым запросом)

### 8.7. rag_complex — тяжёлый поиск

**Файл**: `nodes/rag_complex.py` (85 строк)

"Тяжёлый" путь с повышенными порогами:

| Параметр | rag_simple | rag_complex |
|----------|-----------|-------------|
| top_k | 12 | 60 |
| threshold | 0.45 | 0.50+ |
| min_passages | 5 | 8 |
| min_keyword_overlap | 0.3 | 0.4 |
| rerank | false | true |
| timeout_ms | 250 | 1200 |

Обновляет `plan` на более строгие пороги и выполняет vector search с reranking и MMR.

### 8.8. evaluate_complex — финальная оценка

**Файл**: `nodes/evaluate_complex.py` (70 строк)

Последний шанс найти достаточный контекст. Проверяет 4 источника в порядке приоритета:

1. **Merged passages** — объединение passages из всех attempts через MMR
2. **Last attempt** — только passages из последней попытки (rag_complex)
3. **Fallback passages** — passages сохранённые на этапе triage (fast-path)
4. **Full failure** → abstain

Использует только hard gates (без triage и soft signals).

### 8.9. abstain — честный отказ

**Файл**: `nodes/abstain.py` (59 строк)

Формирует детальное сообщение об отказе с диагностикой:

```
Не удалось найти контекст для: "какие требования СП 1.13130...".
Причины: лучший score (0.312) ниже порога; keyword overlap: active=25% (порог 30%);
LLM-верификатор: passages не релевантны; выполнено 2 переформулировок.
Попыток: 3. Рекомендация: уточните запрос или укажите номер документа.
```

### 8.10. utils — утилиты

**Файл**: `nodes/utils.py` (30 строк)

- `make_retrieval_id(query, filters)` — SHA256-хеш для дедупликации retrieval attempts
- `extract_doc_identifiers(text)` — regex-извлечение номеров нормативных документов (СП, ГОСТ, СНиП, ФЗ, НПБ и т.д.)

---

## 9. Сборка графа

**Файл**: `src/v7/graph.py` (~100 строк)

`build_graph(overrides=None)` — собирает LangGraph StateGraph из всех нод.

```python
def build_graph(overrides=None) -> StateGraph:
    nodes = {
        "intent_gate": intent_gate,
        "router": router,
        "clarify_respond": clarify_respond,
        "rag_simple": rag_simple,
        "evaluate_triage": evaluate_triage,
        "llm_verifier": llm_verifier,
        "rewriter": rewriter,
        "rag_complex": rag_complex,
        "evaluate_complex": evaluate_complex,
        "abstain": abstain,
    }
    if overrides:
        nodes.update(overrides)  # подмена нод для тестов!

    g = StateGraph(RAGState)
    # ... добавление нод и рёбер ...
    return g
```

### Визуальная схема графа

```
                        ┌─────────────┐
                        │ intent_gate │
                        └──────┬──────┘
                               │
                 noise ─── END │ domain
                               │
                        ┌──────▼──────┐
                        │   router    │
                        └──────┬──────┘
                               │
               clarify ── END  │ rag_simple
                               │
                        ┌──────▼──────┐
                        │ rag_simple  │  ← hybrid: vector + BM25 → RRF
                        └──────┬──────┘
                               │
                     ┌─────────▼──────────┐
                     │  evaluate_triage   │
                     └──┬──────┬──────┬───┘
                        │      │      │
              sufficient│  borderline │ clearly_bad
                        │      │      │
                       END  ┌──▼────┐ │
                            │ llm_  │ │
                            │verify │ │
                            └┬──┬──┬┘ │
                             │  │  │  │
                   sufficient│  │  │  │
                             │  │  │  │
                            END │  │  │
                          rewrite│ escalate
                             │  │  │
                        ┌────▼┐ │  │
                        │ re- │ │  │
                        │write│ │  │
                        │ r   │ │  │
                        └──┬──┘ │  │
                           │    │  │
                   (loop)──▶ rag_simple
                                │  │
                         ┌──────▼──▼──────┐
                         │  rag_complex   │  ← rerank + MMR, top_k=60
                         └───────┬────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  evaluate_complex   │
                      └──────┬──────┬───────┘
                             │      │
                   sufficient│      │ failure
                             │      │
                            END  ┌──▼─────┐
                                 │abstain │
                                 └───┬────┘
                                     │
                                    END
```

### Типы рёбер

| Тип | Описание | Пример |
|-----|----------|--------|
| `add_edge(A, B)` | Безусловный переход | `rag_simple → evaluate_triage` |
| `add_conditional_edges(A, fn, map)` | Условный переход | `evaluate_triage → fn → {end/llm_verifier/rag_complex}` |
| `add_edge(A, END)` | Завершение | `abstain → END` |

---

## 10. Мост к инфраструктуре

**Файл**: `src/v7/bridge.py` (~182 строки)

Связывает абстрактный v7 pipeline с конкретной инфраструктурой проекта (ChromaDB, Gemini LLM).

### init_v7_from_chroma(vector_store, llm_provider)

Главная функция инициализации. Вызывается один раз при старте приложения:

```python
def init_v7_from_chroma(vector_store, llm_provider="gemini"):
    # 1. Создать обёртку для ChromaDB → v7 dict format
    search_fn = make_vector_search_fn(vector_store)
    rag_simple_mod.set_vector_search(search_fn)
    rag_complex_mod.set_vector_search(search_fn)

    # 2. Построить BM25 индекс из всего корпуса
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [{"text": doc, "metadata": meta} for doc, meta in zip(...)]
    init_bm25_index(corpus)

    # 3. Инжектировать LLM для verifier и rewriter
    if llm_provider:
        verifier_llm = get_gemini_llm(thinking_budget=1024, ...)
        llm_verifier_mod.set_verify_fn(make_verify_fn(verifier_llm))
        rewriter_llm = get_gemini_llm(thinking_budget=1024)
        rewriter_mod.set_rewrite_fn(make_rewrite_fn(rewriter_llm))
```

### make_vector_search_fn(vector_store)

Конвертирует ChromaDB `similarity_search_with_score` (возвращает L2 distance) в v7 формат (dict со score 0–1):

```python
similarity = 1.0 / (1.0 + distance)  # L2 distance → similarity
```

### make_verify_fn(llm) и make_rewrite_fn(llm)

Создают callable-обёртки для LLM-вызовов, которые инжектируются в ноды.

---

## 11. Полный путь запроса — пошагово

Допустим, пользователь спрашивает: *"Какие требования СП 1.13130 к путям эвакуации?"*

### Шаг 1: Инициализация (один раз при старте)

```python
# app.py
from src.v7.bridge import init_v7_from_chroma
from src.v7.graph import build_graph

init_v7_from_chroma(vector_store)       # инжекция search + LLM
app = build_graph().compile()            # сборка графа
```

### Шаг 2: Запуск графа

```python
result = app.invoke({
    "query": "Какие требования СП 1.13130 к путям эвакуации?",
    "filters": {"doc_type": "СП"},
})
```

### Шаг 3: intent_gate

- Запрос > 3 символов и не в NOISE_QUERIES
- Результат: `intent = "domain"` → переход к `router`

### Шаг 4: router

- Нет маркеров сравнения → `require_multi_doc = False`
- Есть "требования" — не в FACTOID_MARKERS (нет точного совпадения) → default `mmr_lambda = 0.7`
- Создаёт plan: `{top_k: 12, threshold: 0.45, min_passages: 5, ...}`
- Вычисляет `retrieval_id` = SHA256("какие требования сп 1.13130...|{\"doc_type\":\"СП\"}")[:16]

### Шаг 5: rag_simple

- Vector search: 12 passages из ChromaDB
- BM25 search: 12 passages из BM25 индекса
- RRF merge: объединение, top 12 по RRF score
- Записывает RetrievalAttempt(stage="simple")

### Шаг 6: evaluate_triage

- `check_full_triage()`:
  - top_score = 0.72 ≥ 0.45 ✓
  - passage_count = 12 ≥ 5 ✓
  - keyword_overlap = 0.65 ≥ 0.3 ✓
  - max_doc_ratio = 0.5 ≤ 0.8 ✓ (diversity ok)
- Triage = `"sufficient"` → записать `final_passages` → **END**

### Альтернативный путь: borderline

Если бы top_score = 0.38 (ниже threshold, но выше borderline 0.25):
- Triage = `"borderline"` → `llm_verifier`
- LLM говорит: `verdict: "rewrite"`, `rewrite_hint: "добавить ширина коридора"`
- → `rewriter` → новый запрос: "Какие требования СП 1.13130 к путям эвакуации ширина коридора [СП 1.13130]"
- → `rag_simple` (повторный поиск) → `evaluate_triage` → ...

### Альтернативный путь: clearly_bad

Если бы top_score = 0.15:
- Triage = `"clearly_bad"` → `rag_complex`
- Повышенные пороги, top_k=60, rerank=True
- → `evaluate_complex` → merged passages из всех attempts
- Если ничего не помогло → `abstain` → сообщение с диагностикой

---

## 12. Внешние зависимости

### Инфраструктура (src/)

| Модуль | Роль в системе |
|--------|---------------|
| `src/llm_factory.py` | Фабрика LLM: `get_gemini_llm()`, `get_llm()`, `get_vision_llm()` |
| `src/prompt_manager.py` | Jinja2 шаблоны с реестром версий (`prompts/registry.yaml`) |
| `src/vector_store.py` | ChromaDB обёртка |
| `src/agent_tools.py` | Инструменты для ReAct-агентов (search_documents, visual_proof) |
| `src/parsers.py` | Парсинг документов, `extract_text()`, `parse_json_from_response()` |
| `src/semantic_cache.py` | Кеширование запросов |

### Конфигурация

| Файл | Что содержит |
|------|-------------|
| `config/settings.py` | Основные настройки (Settings) — LLM, эмбеддинги, ChromaDB, пороги |
| `config/term_glossary.yaml` | Глоссарий доменных сокращений (СУОТ → система управления охраной труда) |
| `prompts/registry.yaml` | Реестр версий промптов |
| `.env` | Секреты и переопределения настроек |

### UI

`app.py` (Streamlit) — точка входа. Создаёт vector store, инициализирует v7 или multi-agent, рендерит чат.

---

## 13. Тестирование

```bash
source venv/bin/activate
pytest tests/v7/ -v                    # все v7 тесты
pytest tests/v7/ -v -m unit           # только unit
pytest tests/v7/ -v -m integration    # только integration
```

### Структура тестов

```
tests/v7/
├── test_config.py              # V7Config валидация
├── test_state_types.py         # TypedDict контракты
├── test_bridge.py              # Bridge: ChromaDB → v7
├── test_graph.py               # Граф: структура, переходы
├── test_hard_gates.py          # Hard gates, triage
├── test_nlp_core.py            # BM25, RRF, MMR, keywords
└── test_nodes/
    ├── test_intent_gate.py     # Фильтр шума
    ├── test_router.py          # Классификация + plan
    ├── test_rewriter.py        # Переформулировка
    ├── test_evaluate_triage.py # 3-way gate
    ├── test_evaluate_complex.py # Финальная оценка
    ├── test_rag_simple.py      # Hybrid retrieval
    ├── test_llm_verifier.py    # LLM верификация
    ├── test_abstain.py         # Отказ
    └── test_utils.py           # Утилиты
```

### Паттерн тестирования нод

Благодаря dependency injection, ноды тестируются без LLM и без ChromaDB:

```python
def test_rag_simple_returns_attempts():
    # Подменяем vector search стабом
    rag_simple_mod.set_vector_search(lambda **kw: [
        {"text": "...", "score": 0.8, "chunk_id": "c1", "metadata": {}}
    ])

    state = {"query": "тест", "plan": {...}, "retrieval_id": "abc"}
    result = rag_simple(state)

    assert len(result["retrieval_attempts"]) == 1
    assert result["retrieval_attempts"][0]["stage"] == "simple"
```

### Тестирование графа (overrides)

```python
def test_graph_sufficient_path():
    def fake_rag_simple(state):
        return {"retrieval_attempts": [...], "sufficient": True, ...}

    graph = build_graph(overrides={"rag_simple": fake_rag_simple})
    app = graph.compile()
    result = app.invoke({"query": "тест"})
    assert result["sufficient"] is True
```

---

## 14. Как читать и модифицировать код

### С чего начать

1. **`state_types.py`** — прочитать `RAGState`, понять входы и выходы
2. **`graph.py`** — посмотреть на структуру графа (какие ноды, какие переходы)
3. **Любая нода** — понять паттерн read state → call function → write state
4. **`nlp_core.py`** — если нужно понять NLP-логику
5. **`hard_gates.py`** — если нужно понять логику оценки quality
6. **`bridge.py`** — если нужно понять связь с инфраструктурой

### Как добавить новую ноду

1. Создать файл `src/v7/nodes/my_node.py`
2. Написать функцию `def my_node(state: RAGState) -> RAGState:`
3. Написать routing-функцию (если нужны условные переходы)
4. Зарегистрировать в `graph.py` — добавить в dict `nodes` и связать рёбрами
5. Написать тесты в `tests/v7/test_nodes/test_my_node.py`

### Как изменить пороги

Через `.env`:
```bash
V7_HARD_GATE_THRESHOLD=0.80
V7_MIN_KEYWORD_OVERLAP_ACTIVE=0.4
```

Или в `src/v7/config.py` (дефолты).

### Как подключить другой LLM

Изменить `bridge.py` — создать новую функцию `make_verify_fn()` / `make_rewrite_fn()` с нужным провайдером.

### Как отладить

```bash
# Полные логи промптов
DEBUG_PROMPTS=true streamlit run app.py

# LangSmith traces
LANGSMITH_TRACING_V2=true streamlit run app.py

# E2E smoke test
python scripts/verify_ux.py
```
