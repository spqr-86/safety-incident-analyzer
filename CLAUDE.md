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
| 5 | Миграция + cleanup | Pending | — |

### Как продолжить ("продолжи миграцию v7")
1. Прочитай `docs/plans/2026-02-16-v7-migration-design.md` (дизайн + инструкции, секция 7)
2. Найди первый Pending этап в таблице выше
3. Создай ветку `feature/v7-migration-stageN` (или переключись на существующую)
4. Создай implementation plan через `superpowers:writing-plans`
5. Реализуй через `superpowers:subagent-driven-development`
6. Прогони E2E smoke test: `python scripts/verify_ux.py` (удаляет кэш, запускает вопрос через полный пайплайн)
7. Обнови таблицу прогресса **во всех трёх файлах**: `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`

**Синхронизация:** Таблица прогресса дублируется в `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`. При обновлении одного — обнови остальные два.
