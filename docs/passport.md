# Паспорт проекта: AI Safety Compliance Assistant

**Назначение:** Автоматизация проверки нормативной документации (СНиП, ГОСТ, ОТ, ПБ).
**Стек:** LangGraph, Streamlit, ChromaDB, Docling, Google Gemini (`gemini-3-flash-preview`), OpenAI embeddings.

## Особенности архитектуры
- **V7 LangGraph Pipeline (основной):** детерминированный граф состояний без LLM-роутинга —
  `intent_gate → router → rag_simple → evaluate_triage → rag_complex → generate_answer`.
  Hard gates по числовым порогам, явный `abstain` при недостатке данных.
- **Hybrid retrieval:** векторный поиск (ChromaDB) + BM25, RRF-слияние, FlashRank reranking, MMR.
- **Multi-Agent RAG (legacy):** прежний ReAct-агентный пайплайн, заменён V7.

## Результаты
- Eval через `eval/run_v7_eval.py` (golden-датасет, LLM-as-judge): faithfulness > 0.95,
  correctness ~6.9/10 (цель 7.5 — в работе, упирается в баг чанкинга).

## Вызовы (Challenges)
- Баг чанкинга: `_process_docling_document` выроняет пункты норм из индекса.
- Инфляция scores FlashRank в evaluate_complex.
- Run-to-run вариативность LLM-судьи в eval.

## Моя роль (как разработчика)
- Проектирование и реализация V7-графа на LangGraph (тонкие ноды + hard gates).
- Hybrid retrieval, reranking, доменный глоссарий.
- Eval-фреймворк с LLM-as-judge метриками.
- Интеграция визуальных доказательств из PDF.
