# AGENTS.md

This file provides context and instructions for AI agents working in this repository.
The project is an **AI Safety Compliance Assistant** (RAG system for Russian workplace safety regulations).

## 1. Build, Lint, and Test Commands

### Environment Setup
- **Virtual Env:** `python -m venv venv && source venv/bin/activate`
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
  from utils.logging import logger
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
  1.  **Multi-Agent RAG (`agents/multiagent_rag.py`):** Primary. Uses `Router Agent` (LLM), ReAct agent with Parallel Search, and Smart Verifier.
  2.  **Simple RAG (`src/final_chain.py`):** Legacy fallback.
- **Vector Store:** ChromaDB with token-aware batching (`src/vector_store.py`).
- **LLM Factory:** `src/llm_factory.py` unifies provider access (OpenAI, Gemini).

### Prompt Management
- **Templates:** Stored as Jinja2 templates in `prompts/` (e.g., `prompts/agents/research_v1.j2`).
- **Registry:** Mapped in `prompts/registry.yaml`.
- **Versioning:** Controlled via `registry.yaml` or environment variables.

### Key Workflows
- **Term Glossary:** `config/term_glossary.yaml` handles domain abbreviations (e.g., "программа А"). It is applied *before* the router.
- **Visual Proof:** The system extracts images from PDFs (`static/visuals/`) to prove answers.
- **Search Optimization:** Agent search bypasses FlashRank reranking for speed (k=10). BM25 index is cached.

## 4. Cursor / Copilot Rules
*No specific .cursorrules or .github/copilot-instructions.md found. Adhere to the guidelines above.*
