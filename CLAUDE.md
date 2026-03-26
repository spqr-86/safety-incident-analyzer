# CLAUDE.md

## Project Overview

**AI Safety Compliance Assistant** — RAG system for analyzing Russian workplace safety regulations (ГОСТ, СНиП, СП). Uses hybrid retrieval (semantic + BM25), FlashRank reranking, and multi-agent LangGraph workflows.

**Stack**: Python 3.11+, LangChain, LangGraph, ChromaDB, Docling, FlashRank, Streamlit, Google Gemini 3

## Virtual Environment

**IMPORTANT**: Always use the project's venv. Never install packages globally.

```bash
source venv/bin/activate   # before any command
```

## Commands

```bash
streamlit run app.py                             # run app
python index.py                                  # reindex (DESTRUCTIVE: drops ChromaDB)
pytest -v                                        # tests (-m unit / integration / "not slow")
black . && ruff check . --fix                    # lint (run before commits)
python scripts/validate_prompts.py               # validate after prompt changes
```

## Code Style

- **Line length**: 88 (Black/Ruff, see `pyproject.toml`)
- **Imports**: absolute (`from src.utils import logger`). Order: stdlib → third-party → local.
- **Type hints**: `from __future__ import annotations` at top of files.
- **Language**: Domain logic/comments in Russian; infrastructure in English. Follow convention of the file being edited.
- **Testing**: `unittest.mock`, markers `@pytest.mark.unit / integration / slow`, config in `pyproject.toml`.

## Architecture

### Multi-Agent RAG (`agents/multiagent_rag.py`) — primary

```
glossary expansion → regex filter → rag_agent (ReAct) → verifier → format_final
                                   → direct_response (chitchat/out_of_scope)
Revision: verifier (needs_revision) → rag_agent (max 1)
```

- **Regex Filter** (`_classify_query`): deterministic regex, no LLM. Classifies chitchat / out_of_scope / rag.
- **RAG Agent** (flash, thinking: 8192, max 16 steps): ReAct with `search_documents` + `visual_proof` tools.
- **Verifier** (flash, thinking: 1024): JSON fact-check, 6 criteria.
- **Revision**: agent receives previous `draft_answer` + verifier feedback.
- **Shared rules**: `prompts/common/base_rules.j2` — macro imported via `{% import "common/base_rules.j2" as rules %}`.
- **Term Glossary**: `config/term_glossary.yaml` — deterministic expansion of domain abbreviations before query enters the graph. To add terms: edit YAML, no code changes.

**Simple RAG Chain** (`src/final_chain.py`): Legacy fallback. Hybrid retrieval → FlashRank rerank → LLM.

### Prompt Management

Versioned Jinja2 templates. Registry: `prompts/registry.yaml`. Override via env: `PROMPT_RESEARCH_AGENT_VERSION=v2`.

### LLM Factory (`src/llm_factory.py`)

`get_llm()`, `get_gemini_llm(thinking_budget)`, `get_vision_llm()`. Provider via `LLM_PROVIDER` env var. **Note**: Gemini SDK patched to disable AFC (Automatic Function Calling) — prevents duplicate tool calls.

## Common Patterns

### Adding a New Agent
1. Create `prompts/agents/<name>_v1.j2`
2. Register in `prompts/registry.yaml`
3. Add logic in `agents/`
4. Run `python scripts/validate_prompts.py`

### Adding a New Prompt Version
1. Create new file (e.g., `research_v3.j2`)
2. Add to `prompts/registry.yaml`, update `active_version` or test via env override

## Important Notes

- **Russian Language**: System optimized for Russian regulatory text.
- **`index.py` is destructive**: Drops entire ChromaDB before reindexing.
- **Config**: pydantic-settings in `config/settings.py`, copy `.env.example` → `.env`.
- **Debugging**: `DEBUG_PROMPTS=true`, LangSmith traces (`LANGSMITH_TRACING_V2=true`), error reports in `analysis/error_reports/`.
- **Commits**: Do NOT add `Co-Authored-By` lines. Run `black . && ruff check . --fix` before committing.

## Active Work: V7 Migration

**Design:** `docs/plans/2026-02-16-v7-migration-design.md` — полная архитектура, модули, roadmap, инструкции для агента.
**Spec:** `docs/feature/migration-v7` — исходная спецификация v7 (1729 строк, все типы и функции).
**Implementation plans:** `docs/plans/2026-02-16-v7-stage0-implementation.md` — пример формата плана для этапа.

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

**Этап 6 включает** (все на `main`):
- `generate_answer` node + `answer: str` в RAGState — LLM-синтез через bridge DI
- `make_generate_fn()` в bridge (Gemini, thinking_budget=4096, fallback=stub)
- `intent_gate`: regex вместо frozenset (корректная фильтрация шума)
- Тарировка порогов: `HARD_GATE_THRESHOLD=0.50`, `TRIAGE_SOFT_THRESHOLD=0.38` — открыта зона borderline→llm_verifier
- Фикс `top_score` в rag_simple: использует только vector scores (не BM25-inflated)
- E2E smoke test: `python scripts/trace_v7.py` (цветная трассировка пути + answer)
- 151 unit-тест, все зелёные

### E2E smoke test
```bash
python scripts/trace_v7.py "для кого проводится повторный инструктаж?"
python scripts/trace_v7.py --no-chroma "привет как дела"   # stub mode
```

### Известные проблемы (backlog)

**[P1] app.py не читает `result["answer"]`** — самое срочное.
- Сейчас v7 в app.py (строки ~237–261) строит ответ из `final_passages` текстом, игнорируя поле `answer`.
- Нужно добавить ветку `elif result.get("answer"):` перед веткой `final_passages`.
- Затем добавить `USE_V7_GRAPH=true` в `.env` и проверить в Streamlit.

**[P2] FlashRank score inflation в complex path** — evaluate_complex всегда `sufficient`.
- `rag_complex` после реранкинга возвращает FlashRank cross-encoder вероятности (~0.999).
- `evaluate_complex` сравнивает их с `COMPLEX_THRESHOLD=0.35` → всегда проходит, `abstain` никогда не срабатывает.
- Фикс: хранить оригинальный vector score в passages отдельно (`vector_score`), использовать его для threshold check, а FlashRank score — только для сортировки.

**[P3] Integration tests с реальным ChromaDB** — нет coverage на полный стек.
- Текущие тесты: 151 unit-тест с моками. Нет ни одного теста с реальным ChromaDB.
- Нужен `tests/v7/test_integration.py` с `@pytest.mark.integration`, отдельный CI-шаг.

### Как продолжить ("продолжи миграцию v7")

**Шаг 1 — подключить app.py (30 мин):**
1. В `app.py` строки ~237–261, добавить ветку для `result["answer"]` (до `final_passages`).
2. Добавить `USE_V7_GRAPH=true` в `.env`.
3. Запустить `streamlit run app.py` и проверить UI.

**Шаг 2 — починить FlashRank score inflation (1–2 ч):**
1. В `src/v7/nodes/rag_complex.py` — при merge passages сохранять `vector_score` отдельно.
2. В `src/v7/nodes/evaluate_complex.py` — threshold check через `vector_score`, не FlashRank.
3. TDD: написать тест, где FlashRank score высокий, но vector score низкий → ожидать `clearly_bad`.

**Шаг 3 — integration tests (опционально):**
1. Создать `tests/v7/test_integration.py`, маркер `@pytest.mark.integration`.
2. Требует реальный ChromaDB (не мок), запускать отдельно: `pytest -m integration`.

**Синхронизация:** Таблица прогресса дублируется в `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`. При обновлении одного — обнови остальные два.
