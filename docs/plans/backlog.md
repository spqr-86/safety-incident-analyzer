# Backlog

## План улучшений RAG (pipeline quality) — 2026-05-15

Приоритизированный список после eval correctness=6.69 (цель 7.5):

| # | Улучшение | Сложность | Ожидаемый эффект | Статус |
|---|-----------|-----------|------------------|--------|
| 1 | **Regex URL/date noise cleanup** в `_clean_noise` (file_handler.py) | ~30 мин | Программы А/Б/В 0→? | ✅ DONE 2026-05-16 |
| 2 | **Contextual retrieval** — LLM генерирует 1-2 предложения контекста на чанк перед embedding | ~2 дня | +35-49% recall (Anthropic) |  |
| 3 | **Parent-context chunking** — small chunks для поиска, large для генерации | ~1 день | +coherence ответа |  |
| 4 | **Overlap 10-15%** — добавить перекрытие чанков в grouping логике | ~1 ч | +continuity |  |
| 5 | **Hybrid retriever fix** — RRF вместо concatenation в applicability_retriever.py | ~2 ч | +recall на BM25-only вопросах |  |

---

## ~~[P1] Баг чанкинга: `_process_docling_document` выроняет пункты норм~~ ✅ FIXED 2026-05-15

Root cause: `MIN_BBOX_HEIGHT=7` отбрасывал item целиком (с текстом) до группировки.
Короткие однострочные пункты ППРФ 2464 имели bbox_h~5.7, попадали под фильтр.

Исправлено в commit 32c82ae:
- `MIN_BBOX_HEIGHT` фильтр зануляет bbox, но оставляет текст в индексе
- `MAX_CHUNK_SIZE` теперь берётся из `settings.CHUNK_SIZE` (1500)
- `update_bbox=False` (смена страницы) flush-ит чанк
- `PIPELINE_VERSION` → v2.2-grouped
- 830 → 1069 чанков после переиндексации (+239)
- Регресс-тест: `tests/test_docling_structure.py::test_short_bbox_keeps_text_drops_bbox`

---

## [P1] app.py не читает `result["answer"]`

Сейчас v7 в app.py (строки ~237–261) строит ответ из `final_passages` текстом, игнорируя поле `answer`.

**Фикс:**
1. Добавить ветку `elif result.get("answer"):` перед веткой `final_passages` в app.py.
2. Добавить `USE_V7_GRAPH=true` в `.env`.
3. Проверить в Streamlit.

---

## [P2] FlashRank score inflation в evaluate_complex

`rag_complex` после реранкинга возвращает FlashRank cross-encoder вероятности (~0.999).
`evaluate_complex` сравнивает их с `COMPLEX_THRESHOLD=0.35` → всегда `sufficient`, `abstain` никогда не срабатывает.

**Фикс:**
1. В `src/v7/nodes/rag_complex.py` — при merge passages сохранять `vector_score` отдельно.
2. В `src/v7/nodes/evaluate_complex.py` — threshold check через `vector_score`, FlashRank score — только для сортировки.
3. TDD: тест где FlashRank score высокий, но vector score низкий → ожидать `clearly_bad`.

**Влияние:** false_sufficiency_rate завышена до этого фикса.

---

## [P3] Integration tests с реальным ChromaDB

Текущие тесты: unit с моками. Нет ни одного теста с реальным ChromaDB.

**Фикс:**
1. Создать `tests/v7/test_integration.py`, маркер `@pytest.mark.integration`.
2. Запускать отдельно: `pytest -m integration`.

---

## ~~[P0] Retry при Gemini 503 в bridge.py~~ ✅ DONE 2026-05-08

`make_generate_fn()` в `src/v7/bridge.py` при 503 от Gemini сразу падает в stub и возвращает сырой текст чанков вместо синтезированного ответа.

**Фикс:**
1. В `make_generate_fn` обернуть вызов Gemini в retry с экспоненциальной задержкой (tenacity или ручной loop).
2. Параметры: 3 попытки, задержка 2→4→8 сек.
3. Логировать каждую попытку через structlog.
4. Stub как финальный fallback только после всех ретраев.

```python
# примерный паттерн
from tenacity import retry, stop_after_attempt, wait_exponential
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def _call_gemini(...): ...
```

---

## [P4] MCP-сервер поверх RAG (knowledge search для Claude)

Сделать MCP-сервер чтобы Claude мог искать по ~/knowledge/ как инструментом.

**Компоненты:**
1. Индексатор: читает ~/knowledge/ (md напрямую, PDF/DOCX через Docling) → отдельная ChromaDB коллекция `knowledge_base`
2. `mcp_rag_server.py`: tool `search_knowledge(query, top_k=5)` → гибридный поиск (vector + BM25) из src/v7/
3. Регистрация в ~/.claude/settings.json как MCP server

**Результат:** Claude видит search_knowledge как инструмент, вызывает сам без Bash костыля.
