# Паспорт проекта: AI Safety Compliance Assistant

**Назначение:** Автоматизация проверки нормативной документации (СНиП, ГОСТ, ОТ, ПБ).
**Стек:** LangGraph, Streamlit, ChromaDB, Gemini 2.0 Flash.

## Особенности архитектуры
- **Multi-Agent RAG Workflow:**
    - Маршрутизация запросов (Router) без LLM.
    - Агент (ReAct) с инструментами поиска.
    - Верификатор (Smart Verifier) для проверки фактов.
- **V7 Graph:** Модульный pipeline для сложных запросов.

## Результаты
- Faithfulness: 89%
- Citation Rate: 95%

## Вызовы (Challenges)
- Инфляция scores (FlashRank).
- Интеграция V7 графа.
- Coverage тестов (в процессе).

## Моя роль (как разработчика)
- Реализация Multi-Agent архитектуры на LangGraph.
- Оптимизация RAG-цепочки.
- Интеграция визуальных доказательств из PDF.
