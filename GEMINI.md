# Project Overview

This project is an AI Safety Compliance Assistant, a RAG (Retrieval-Augmented Generation) system designed to analyze regulatory documents related to labor protection. It is built with Python and utilizes a multi-agent approach with LangGraph to ensure the quality of its responses.

The core technologies used are:
- **Python 3.11+**
- **Streamlit** for the user interface.
- **LangChain** and **LangGraph** for the LLM framework.
- **Google Gemini 3** as LLM provider.
- **ChromaDB** as the vector store.
- **Docling** for ETL (Extract, Transform, Load).
- **FlashRank** for reranking.
- **Ragas** and custom metrics for evaluation.

The system features a hybrid search combining semantic (vector) and keyword-based (BM25) search, smart ranking with FlashRank, and a multi-level verification process using agents with Chain-of-Thought (CoT) reasoning. It also includes a domain-specific glossary to expand queries with official terminology.

# Building and Running

## Local Setup

**Prerequisites:**
- Python 3.11+
- API keys for OpenAI and/or Gemini

**1. Clone and install dependencies:**
```bash
git clone https://github.com/spqr-86/safety-incident-analyzer.git
cd safety-incident-analyzer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Rule:** Always use the virtual environment (`venv`) for all Python commands. Never install packages globally.

**2. Configure the environment:**
Create a `.env` file in the project root (you can use `.env.example` as a template):
```env
# LLM Provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

# Embeddings
EMBEDDING_PROVIDER=openai # or hf_api, local
```

**3. Index documents:**
Place your documents (PDF, DOCX, MD) in the `source_docs/` directory and run:
```bash
python index.py
```

**4. Run the application:**
```bash
streamlit run app.py
```

## Testing

The project uses `pytest` for testing. To run the tests, execute the following command in the project root:
```bash
pytest
```

# Development Conventions

- **Code Formatting:** The project uses **Black** for code formatting with a line length of 88 characters.
- **Linting:** **Ruff** is used for linting, checking for errors (E), fatal errors (F), and warnings (W). The E501 error (line too long) is ignored as it is handled by Black.
- **Testing:** The project uses **pytest** for testing. Tests are located in the `tests/` directory and follow the `test_*.py` naming convention. `unittest.mock` is used for mocking dependencies.
- **Prompt Management:** Prompts are managed using Jinja2 templates stored in the `prompts/` directory. A `registry.yaml` file in the same directory controls the active versions of the prompts.
- **Commits:** Do NOT add `Co-Authored-By` lines. Run `black . && ruff check . --fix` before committing.
- **Architecture:** The project follows a multi-agent RAG architecture orchestrated by LangGraph. The core logic is located in the `src/` and `agents/` directories. The configuration is managed in the `config/` directory. Detailed documentation about the architecture can be found in `docs/architecture/README.md`.

# Active Work: V7 Migration

**Синхронизация:** Таблица прогресса дублируется в `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`. При обновлении одного — обнови остальные два.

**Design:** `docs/plans/2026-02-16-v7-migration-design.md` — полная архитектура, модули, roadmap, инструкции для агента (секция 7).
**Spec:** `docs/feature/migration-v7` — исходная спецификация v7 (1729 строк, все типы и функции).
**Plan example:** `docs/plans/2026-02-16-v7-stage0-implementation.md` — пример формата плана для этапа.

### Ключевые принципы v7 (секция 5 дизайн-документа)
- **5.5 Не изобретать велосипед** — использовать библиотеки (`rank_bm25`, `pymorphy3`, `razdel`, `pydantic`), не писать свои реализации
- **5.6 Строгий граф зависимостей** — `state_types ← config ← nlp_core ← hard_gates ← nodes ← graph`, обратных импортов нет
- **5.7 Тонкие ноды** — ноды только оркестрируют (read state → call function → write state), вся логика в `nlp_core`/`hard_gates`

## Прогресс
| Этап | Модуль | Статус | Ветка |
|------|--------|--------|-------|
| 0 | `state_types` + `config_v7` | ✅ Done | `feature/v7-migration-stage0` |
| 1 | `nlp_core` | ✅ Done | `feature/v7-migration-stage1` |
| 2 | `hard_gates` | Pending | — |
| 3 | `nodes/*` | Pending | — |
| 4 | `graph.py` | Pending | — |
| 5 | Миграция + cleanup | Pending | — |

## Как продолжить ("продолжи миграцию v7")
1. Прочитай `docs/plans/2026-02-16-v7-migration-design.md` (дизайн + инструкции, секция 7)
2. Найди первый Pending этап в таблице выше
3. Создай ветку `feature/v7-migration-stageN` (или переключись на существующую)
4. Создай implementation plan
5. Реализуй
6. Прогони E2E smoke test: `python scripts/verify_ux.py` (удаляет кэш, запускает вопрос через полный пайплайн)
7. Обнови таблицу прогресса **во всех трёх файлах**: `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`
