# AGENTS.md

This file provides context and instructions for AI agents working in this repository.
The project is an **AI Safety Compliance Assistant** (RAG system for Russian workplace safety regulations).

## 1. Build, Lint, and Test Commands

### Environment Setup
- **Virtual Env:** `python -m venv venv && source venv/bin/activate`
- **Rule:** Always use the virtual environment (`venv`) for all Python commands.
- **Dependencies:** `pip install -r requirements.txt`
- **Configuration:** Copy `.env.example` to `.env` and populate API keys.

### Running the Application
- Start the Streamlit app:
  ```bash
  streamlit run app.py
  ```

### Linting and Formatting
- **Format Code (Black):**
  ```bash
  black .
  ```
- **Lint Code (Ruff):**
  ```bash
  ruff check . --fix
  ```
- **Rule:** Always run `black` and `ruff` before committing changes. Ensure no linting errors remain.

### Testing
The project uses `pytest` with configuration in `pyproject.toml`.

- **Run all tests:**
  ```bash
  pytest -v
  ```
- **Run unit tests only:**
  ```bash
  pytest -m unit
  ```
- **Run integration tests only:**
  ```bash
  pytest -m integration
  ```
- **Skip slow tests:**
  ```bash
  pytest -m "not slow"
  ```
- **Run a single test file (with stdout):**
  ```bash
  pytest tests/test_agent_tools.py -v -s
  ```
- **Run a specific test function:**
  ```bash
  pytest tests/test_file.py::test_function_name -v
  ```

**Testing Rules:**
1.  **Mocking:** Use `unittest.mock` for unit tests to avoid external API calls (LLM, Vector DB).
2.  **Markers:** Use `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow` appropriately.
3.  **No `conftest.py`:** Note that tests currently use `unittest.mock` and do not rely on a global `conftest.py`.

### Domain-Specific Commands
- **Index Documents (DESTRUCTIVE):**
  ```bash
  python index.py
  ```
  *Warning: This deletes the existing ChromaDB before re-indexing.*

- **Validate Prompts:**
  ```bash
  python scripts/validate_prompts.py
  ```
  *Rule: Run this after any modification to prompt templates.*

- **Evaluation:**
  ```bash
  python eval/run_full_evaluation.py
  ```

## 2. Code Style & Conventions

### General Style
- **Python Version:** 3.11+
- **Line Length:** 88 characters (enforced by Black/Ruff).
- **Type Hints:** Mandatory for function arguments and return values.
  - Use `from __future__ import annotations` at the top of files.
  - Example: `def process_data(items: list[str]) -> dict[str, Any]:`
- **Imports:**
  - Grouping: Standard library, Third-party, Local application (src).
  - Use absolute imports for project modules: `from src.utils import logger` instead of relative `from ..utils import logger`.

### Naming Conventions
- **Variables & Functions:** `snake_case` (e.g., `calculate_metrics`, `user_input`)
- **Classes:** `PascalCase` (e.g., `RAGAgent`, `DocumentProcessor`)
- **Constants:** `UPPER_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_MODEL`)
- **Private Members:** `_leading_underscore` (e.g., `_sanitize_metadata`)

### Documentation & Comments
- **Docstrings:** Required for complex public functions and classes.
- **Language:** The codebase handles Russian regulatory text.
  - Domain-specific logic/comments may be in Russian (as seen in `src/vector_store.py`).
  - General infrastructure code often uses English.
  - **Rule:** Follow the language convention of the file you are editing.

### Error Handling
- Use specific exception types (e.g., `ValueError`, `ConnectionError`) rather than bare `Exception`.
- **LLM/API Calls:** Wrap external calls in `try...except` blocks. Log errors using the project's logger.
  ```python
  from src.utils import logger
  try:
      response = llm.invoke(...)
  except Exception as e:
      logger.error(f"LLM call failed: {e}")
      raise
  ```

## 3. Architecture & Project Structure

### Core Components
- **Frameworks:** LangChain, LangGraph, Streamlit.
- **RAG Approaches:**
  1.  **Multi-Agent RAG (`agents/multiagent_rag.py`):** Primary. Uses deterministic regex router (no LLM), ReAct agent with search tools, and Smart Verifier.
  2.  **Simple RAG (`src/final_chain.py`):** Legacy fallback.
- **Vector Store:** ChromaDB with token-aware batching (`src/vector_store.py`).
- **LLM Factory:** `src/llm_factory.py` unifies provider access (OpenAI, Gemini).

### Prompt Management
- **Templates:** Stored as Jinja2 templates in `prompts/` (e.g., `prompts/agents/research_v1.j2`).
- **Registry:** Mapped in `prompts/registry.yaml`.
- **Versioning:** Controlled via `registry.yaml` or environment variables.

### Key Workflows
- **Term Glossary:** `config/term_glossary.yaml` handles domain abbreviations (e.g., "программа А"). Expansion logic lives in `src/glossary.py` (`expand_query_with_glossary`). Applied in the V7 `router` node (writes the expanded `active_query`); the legacy `multiagent_rag` applies it before its router.
- **Visual Proof:** The system extracts images from PDFs (`static/visuals/`) to prove answers.
- **Search Optimization:** Agent search bypasses FlashRank reranking for speed (k=10). BM25 index is cached.

## 4. Active Work: V7 Migration

**Синхронизация:** Таблица прогресса дублируется в `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`. При обновлении одного — обнови остальные два.

**Design:** `docs/plans/2026-02-16-v7-migration-design.md` — полная архитектура, модули, roadmap, инструкции для агента (секция 7).
**Spec:** `docs/feature/migration-v7` — исходная спецификация v7 (1729 строк, все типы и функции).
**Plan example:** `docs/plans/2026-02-16-v7-stage0-implementation.md` — пример формата плана для этапа.

### Ключевые принципы v7 (секция 5 дизайн-документа)
- **5.5 Не изобретать велосипед** — использовать библиотеки (`rank_bm25`, `pymorphy3`, `razdel`, `pydantic`), не писать свои реализации
- **5.6 Строгий граф зависимостей** — `state_types ← config ← nlp_core ← hard_gates ← nodes ← graph`, обратных импортов нет
- **5.7 Тонкие ноды** — ноды только оркестрируют (read state → call function → write state), вся логика в `nlp_core`/`hard_gates`

### Прогресс
| Этап | Модуль | Статус | Ветка |
|------|--------|--------|-------|
| 0 | `state_types` + `config_v7` | ✅ Done | `feature/v7-migration-stage0` |
| 1 | `nlp_core` | ✅ Done | `feature/v7-migration-stage1` |
| 2 | `hard_gates` | ✅ Done | `feature/v7-migration-stage1` |
| 3 | `nodes/*` | ✅ Done | `feature/v7-migration-stage1` |
| 4 | `graph.py` | ✅ Done | `feature/v7-migration-stage1` |
| 5 | Миграция + cleanup | ✅ Done | `feature/v7-migration-stage1` |
| 6 | Production readiness | ✅ Done | `main` |

### Как продолжить ("продолжи миграцию v7")

Все этапы 0–6 Done. Backlog (подробнее в `CLAUDE.md`):

**[P1] Подключить app.py к `result["answer"]`** — срочно.
- `app.py` строки ~237–261: добавить `elif result.get("answer"):` перед веткой `final_passages`.
- Добавить `USE_V7_GRAPH=true` в `.env`, проверить через `streamlit run app.py`.

**[P2] FlashRank score inflation** — evaluate_complex никогда не abstain.
- После реранкинга scores ~0.999, COMPLEX_THRESHOLD=0.35 → всегда `sufficient`.
- Фикс: хранить `vector_score` в passages, использовать для threshold; FlashRank только для сортировки.

**[P3] Integration tests** — нет coverage с реальным ChromaDB.
- Создать `tests/v7/test_integration.py`, маркер `@pytest.mark.integration`.

E2E smoke test: `python scripts/trace_v7.py "вопрос"`

## 5. Cursor / Copilot Rules
*No specific .cursorrules or .github/copilot-instructions.md found. Adhere to the guidelines above.*
