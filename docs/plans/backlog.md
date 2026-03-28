# Backlog

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
