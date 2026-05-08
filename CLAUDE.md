# CLAUDE.md

## Project Overview

**AI Safety Compliance Assistant** — RAG system for analyzing Russian workplace safety regulations (ГОСТ, СНиП, СП). Uses hybrid retrieval (semantic + BM25), FlashRank reranking, and multi-agent LangGraph workflows.

**Stack**: Python 3.11+, LangChain, LangGraph, ChromaDB, Docling, FlashRank, Streamlit, Google Gemini 3

## Virtual Environment

**IMPORTANT**: Always use the project's venv. Never install packages globally.

```bash
source venv/bin/activate   # before any command
```

## Deployment (VPS)

- **URL:** http://213.176.64.237:8502
- **Port:** 8502 (UFW opened), tmux session `sia` (attach: `tmux a -t sia`)
- **Start:** `cd /home/petr/projects/safety-incident-analyzer && source venv/bin/activate && streamlit run app.py --server.port 8502`
- **Indexed docs:** 7 PDFs (2464н, ТК РФ, СОУТ, ПБ и др.) → 749 chunks, collection `documents` в `chroma_db_openai/`

**VPS env requirements:**
- `HTTPS_PROXY=socks5h://localhost:40000` + `HTTP_PROXY=socks5h://localhost:40000` — WARP proxy для Gemini (VPS AEZA заблокирован по ASN)
- PySocks установлен (`pip install httpx[socks] requests[socks]`) — нужен для SOCKS в httpx/requests

## Known Issues

- ~~**Gemini 503 → stub fallback**~~ ✅ Fixed 2026-05-08: tenacity retry (3 attempts, exp backoff 2→4→8s), fallback to stub only after all retries.
- **P1** `app.py` не читает `result["answer"]` — см. backlog.md
- **P2** FlashRank score inflation в evaluate_complex — см. backlog.md

## Commands

```bash
streamlit run app.py --server.port 8502          # run app (VPS port)
python index.py                                  # reindex (DESTRUCTIVE: drops ChromaDB)
pytest -v                                        # tests (-m unit / integration / "not slow")
black . && ruff check . --fix                    # lint (run before commits)
python scripts/validate_prompts.py               # validate after prompt changes
python scripts/trace_v7.py "вопрос"             # E2E smoke test с трассировкой
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

### Backlog

См. `docs/plans/backlog.md`.

**Синхронизация:** Таблица прогресса дублируется в `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`. При обновлении одного — обнови остальные два.

---

## Session Log

### 2026-05-08

- **Сделано:** visual_enrichment нода добавлена в V7 pipeline (src/v7/nodes/visual_enrichment.py). Вставлена между evaluate_complex/triage/verifier → generate_answer в graph.py. make_visual_proof_fn() + inject в bridge.py. 270 тестов, все зелёные.
- **Решения:** Ruff-хук снимает неиспользуемые импорты после каждого Edit — нужно добавлять import и его использование в одном edit. Обходное решение: сначала добавить использование, потом import.
- **Наблюдения:** visual_enrichment — no-op без inject (нет visual_proof_fn) и нет agent_tools на VPS → безопасно деплоить.

### 2026-05-07

- **Сделано:** Запущено на VPS порт 8502 (UFW opened). Переиндексировано 7 PDF → 749 чанков (ранее был только 1 документ — кеш из failed SOCKS runs). Исправлен `evaluate_complex`: `top_k` 12 → 24 (ответы стали полными, 8 категорий вместо 2-3 для "программа А"). Установлен PySocks для WARP proxy в Docling. В `.env` прописаны `HTTP(S)_PROXY=socks5h://localhost:40000`.
- **Решения:** WARP proxy через `.env` (а не setenv в процессе) — Docling читает `.env` при старте. ChromaDB `document_cache/*.pkl` был corrupted (0 chunks) — удалили, переиндексировали. Gemini model = `gemini-3-flash-preview` (подтверждено через API list).
- **Наблюдения:** Gemini 503 при перегрузке → bridge.py падает в stub → сырой текст вместо ответа. Нужен retry. `trace_v7.py` показывает `fallback_passages: 12` — это NOT означает использование fallback, это поле существует в state (от rag_simple), misleading.
