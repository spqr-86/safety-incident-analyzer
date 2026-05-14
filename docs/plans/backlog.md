# Backlog

## [P1] Баг чанкинга: `_process_docling_document` выроняет пункты норм

`src/file_handler.py` (`_process_docling_document`) при группировке чанков теряет
целые элементы документа. Подтверждено: «Повторный инструктаж по охране труда
проводится не реже одного раза в 6 месяцев» (п. 14 Приказа 2464) есть в исходном
PDF и чисто извлекается Docling (`export_to_markdown`), но в индексе ChromaDB
отсутствует — единственный чанк с «повторный инструктаж» содержит только п. 15.

**Влияние:** напрямую ограничивает eval correctness — нормы, которых нет в
индексе, дают честный «в фрагментах ответа нет» (faithfulness 1.0, correctness 0).

**Фикс:**
1. Дебаг логики группировки/`update_bbox` в `_process_docling_document` —
   подозрение на обработку смены страницы (`return False` при page change) или
   dedup по `seen_chunk_hashes`.
2. Сравнить кол-во элементов `doc.texts` с кол-вом финальных чанков на 2464.pdf.
3. После фикса — переиндексация (`python index.py` — DESTRUCTIVE, дропает ChromaDB).
4. Регресс-проверка: чанк с «не реже одного раза в 6 месяцев» должен быть в индексе.

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
