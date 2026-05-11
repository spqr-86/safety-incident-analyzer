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
- **Indexed docs:** 8 PDFs (2464н, ТК РФ, СОУТ, ПБ, 776н и др.) → 830 chunks, collection `documents` в `chroma_db_openai/`

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

### 2026-05-11 (сессия 14)

- **Сделано:** Eval на gemini-2.5-flash: V7=0.614, V8_promptv2=0.611. Промпт v2 нейтрален по completeness но снизил false_abstain 0.057→0.029 (-50%). FastAPI api.py создан (порт 8503, POST /query + GET /health, lifespan init). fastapi+uvicorn+structlog добавлены в requirements.txt. V8 Epic 4 план сохранён: docs/plans/2026-05-11-eval-improvements.md.
- **Решения:** completeness 0.61 = потолок corpus coverage (нет Приказа 223н по НС, нет КоАП). Eval через леммы не ловит инверсии — нужен negative_keywords. Citation validity не измеряется — разрыв между промптом v2 и eval.
- **Наблюдения:** OOS detection = 1.0 (идеально). False abstain = 0.029 (хорошо). Система работает, bottleneck — corpus и eval metrics, не retrieval/generation.

### 2026-05-09 (сессия 13)

- **Сделано:** V8 Epic 3 Multi-Query Expand (make_expand_fn, set_expand_fn DI, RRF слияние, 7 тестов). Eval: +0.000 completeness — retrieval не bottleneck. Добавлен 776н (СУОТ) в corpus, переиндексировано 749→830 чанков, completeness 0.567→0.589. Датасет очищен 38→36 (удалены Q34 ПБ склад, Q35 вентиляция как OOS). Промпт генерации v2: 14 проблем исправлено (роль убрана, 3 варианта ответа, HIGH/MED/LOW score, дословные цитаты, обязательные ссылки, противоречия без приоритизации). Decision log: docs/plans/2026-05-09-generate-prompt-v2.md.
- **Решения:** Retrieval работает (16+ фрагментов находятся) — bottleneck в generation. Multi-query expand нейтрален на stub LLM. Corpus gaps важнее всего — 776н дал +0.023. Роль "эксперта" в промпте провоцирует достройку из общих знаний — заменена на "отвечаешь СТРОГО на основе фрагментов".
- **Наблюдения:** gemini-3-flash → 404 (нестабильна). Все eval — на stub. Нужна gemini-2.5-flash для честного eval. Промпт v2 не оценён с реальным LLM. Corpus: 830 чанков, dataset: 36 вопросов.

### 2026-05-08 (сессия 12)

- **Сделано:** V8 Epic 1 (eval suite: metrics.py, run_eval.py, compare.py, 29 тестов) + V8 Epic 2 (evidence_assess: 3-verdict флаг, FlashRank в rag_simple, 314 тестов). Датасет очищен: 49→38 вопросов (удалены out-of-corpus). Модель: gemini-3-flash (нестабильна). V7 baseline stub-загрязнён — нужен перегон.
- **Решения:** Смена модели в .env = невалидация baseline. Перед сменой модели предупреждать пользователя. Gemini dashboard "Gemini 3 Flash" ≠ API ID `gemini-3-flash` (404). Работающий: `gemini-3-flash-preview` (20 RPD) или `gemini-2.5-flash` (10K RPD).
- **Наблюдения:** completeness=0.49 на 49 вопросах — 11 вопросов были out-of-corpus → реальный baseline на 38 вопросах ожидается 0.60-0.65. V8 evidence_assess даёт +0.002 — нужен Epic 3 (Multi-Query Expand) для реального улучшения.

### 2026-05-08 (сессия 11)

- **Сделано:** visual_enrichment нода добавлена в V7 pipeline. make_visual_proof_fn() + inject в bridge.py. 270 тестов, все зелёные.
- **Решения:** Ruff-хук снимает неиспользуемые импорты после каждого Edit — добавлять import и использование в одном edit.
- **Наблюдения:** visual_enrichment — no-op без inject → безопасно деплоить.

### 2026-05-07

- **Сделано:** Запущено на VPS порт 8502 (UFW opened). Переиндексировано 7 PDF → 749 чанков (ранее был только 1 документ — кеш из failed SOCKS runs). Исправлен `evaluate_complex`: `top_k` 12 → 24 (ответы стали полными, 8 категорий вместо 2-3 для "программа А"). Установлен PySocks для WARP proxy в Docling. В `.env` прописаны `HTTP(S)_PROXY=socks5h://localhost:40000`.
- **Решения:** WARP proxy через `.env` (а не setenv в процессе) — Docling читает `.env` при старте. ChromaDB `document_cache/*.pkl` был corrupted (0 chunks) — удалили, переиндексировали. Gemini model = `gemini-3-flash-preview` (подтверждено через API list).
- **Наблюдения:** Gemini 503 при перегрузке → bridge.py падает в stub → сырой текст вместо ответа. Нужен retry. `trace_v7.py` показывает `fallback_passages: 12` — это NOT означает использование fallback, это поле существует в state (от rag_simple), misleading.
