# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Safety Compliance Assistant** — RAG system for analyzing Russian workplace safety regulations (ГОСТ, СНиП, СП). Uses hybrid retrieval (semantic + BM25), FlashRank reranking, and multi-agent LangGraph workflows for quality control.

**Stack**: Python 3.11+, LangChain, LangGraph, ChromaDB, Docling, FlashRank, Streamlit, Google Gemini 3

## Virtual Environment

**IMPORTANT**: Always use the project's virtual environment for all operations (running scripts, installing packages, executing tests, etc.). Never install packages into the global Python.

```bash
# Activate before any command
source venv/bin/activate

# Or use the venv python/pip directly
./venv/bin/python ...
./venv/bin/pip install ...
```

## Development Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys

# Run app
streamlit run app.py

# Index documents (DESTRUCTIVE: deletes existing ChromaDB first)
python index.py

# Tests
pytest -v                              # all tests
pytest -m unit                         # unit only
pytest -m integration                  # integration only
pytest -m "not slow"                   # skip slow
pytest tests/test_agent_tools.py -v -s # single file, with stdout

# Lint/format
black .
ruff check . --fix

# Prompt validation (run after any prompt changes)
python scripts/validate_prompts.py

# Evaluation (requires LANGSMITH_API_KEY)
python eval/run_full_evaluation.py
python scripts/check_target_metrics.py
python scripts/compare_with_baseline.py
```

## Architecture

### Two RAG Approaches (coexist, selectable in UI)

1. **Multi-Agent RAG** (`agents/multiagent_rag.py`): Primary approach using Gemini/OpenAI with a ReAct agent and thinking levels.
   ```
   glossary expansion → regex filter → rag_agent (ReAct) → verifier → format_final
                                      → direct_response (chitchat/out_of_scope)

   Revision: verifier (needs_revision) → rag_agent (max 1)
   ```
   - **Regex Filter** (`_classify_query`): No LLM — regex patterns classify chitchat / out_of_scope / rag. Fast, deterministic, zero-cost.
   - **RAG Agent** (flash, thinking: 8192): Single ReAct agent with `search_documents` + `visual_proof` tools. Handles both simple and complex queries via conditional decomposition (few-shot examples in prompt).
   - **Verifier** (flash, thinking: 1024): JSON fact-check (approved / needs_revision), 6 criteria
   - **Revision with context**: On needs_revision, agent receives previous `draft_answer` + verifier feedback for targeted fixes
   - **Shared rules**: `prompts/common/base_rules.j2` macro imported by RAG agent prompt
   - **Term Glossary**: `config/term_glossary.yaml` — deterministic expansion of domain abbreviations (e.g., "программа А" → official term) before query enters the graph. Handles Russian morphology via stem-based matching.

2. **Simple RAG Chain** (`src/final_chain.py`): Legacy fallback. Hybrid retrieval → FlashRank rerank (top_n=12) → LLM generation. Uses `ApplicabilityRetriever` for LLM-powered query expansion.

### Tools (`src/agent_tools.py`)

- **`search_documents`**: Hybrid retrieval + Smart Context Extension (sliding window ±2 chunks with range merging and deduplication by source+chunk_id)
- **`visual_proof`**: Two modes — `mode="show"` returns cropped PDF image, `mode="analyze"` sends full page with red box to VLM for OCR/table extraction

### Key Subsystems

**Prompt Management**: All prompts are versioned Jinja2 templates. Registry at `prompts/registry.yaml` maps IDs to files. Override versions via env: `PROMPT_RESEARCH_AGENT_VERSION=v2`. Debug with `DEBUG_PROMPTS=true`.

**LLM Factory** (`src/llm_factory.py`): Unified interface — `get_llm()` (OpenAI), `get_gemini_llm(thinking_budget, response_mime_type)`, `get_vision_llm()`. Provider set via `LLM_PROVIDER` env var.

**Document Processing** (`src/file_handler.py`): Docling PDF/DOCX→MD conversion with MD5 caching (7-day expiry), BBox extraction for visual proof, token-aware batching.

**Vector Store** (`src/vector_store.py`): ChromaDB with token-aware batching (280K token limit per batch for OpenAI). Supports OpenAI, HuggingFace API, or local embeddings.

## Configuration

Copy `.env.example` → `.env`. Key settings:

| Setting | Default | Notes |
|---------|---------|-------|
| `LLM_PROVIDER` | `openai` | `openai` |
| `EMBEDDING_PROVIDER` | `openai` | `openai`, `hf_api`, or `local` |
| `CHUNK_SIZE` | 1500 | |
| `CHUNK_OVERLAP` | 400 | |
| `VECTOR_SEARCH_K` | 10 | |
| `HYBRID_RETRIEVER_WEIGHTS` | [0.6, 0.4] | [semantic, keyword] |
| `GEMINI_FAST_MODEL` | `gemini-3-flash-preview` | RAG Agent + Verifier |

All config loaded via pydantic-settings in `config/settings.py`. ChromaDB path auto-appends provider suffix (e.g., `chroma_db_openai`).

## Common Patterns

### Adding a New Agent

1. Create prompt template in `prompts/agents/<name>_v1.j2`
2. Register in `prompts/registry.yaml` with active_version
3. Add agent logic in `agents/` or as a node in existing workflow
4. Validate: `python scripts/validate_prompts.py`

### Adding a New Prompt Version

1. Create new template file (e.g., `research_v3.j2`)
2. Add version entry in `prompts/registry.yaml`
3. Update `active_version` or test via env override: `PROMPT_RESEARCH_AGENT_VERSION=v3`

## Important Notes

- **Russian Language**: System optimized for Russian regulatory text.
- **`index.py` is destructive**: Deletes entire ChromaDB before reindexing. No incremental mode.
- **BASE_RULES**: `prompts/common/base_rules.j2` macro enforces strict factual adherence — imported by RAG agent prompt. 10 edge cases including glossary integration and unknown abbreviation fallback.
- **Term Glossary**: `config/term_glossary.yaml` maps unofficial domain abbreviations to official terms. Loaded once at workflow init, applied deterministically before regex filter. To add terms: edit the YAML file, no code changes needed.
- **Visual Proof images**: Saved to `static/visuals/` as `proof_<md5[:8]>.png`. Requires `pymupdf` and `PIL`.
- **Gemini Rate Limits**: Free tier has strict quotas. Pass `llm_provider="openai"` to `MultiAgentRAGWorkflow` as fallback.
- **FlashRank**: CPU-intensive, expect 2-3s reranking latency.
- **Streamlit Cloud**: `app.py` conditionally patches sqlite3 with pysqlite3-binary for deployment compatibility.
- **Test config**: In `pyproject.toml` (not pytest.ini). Markers: `slow`, `integration`, `unit`. Tests use `unittest.mock`, no conftest.py.
- **Debugging**: Check `analysis/error_reports/`, enable `DEBUG_PROMPTS=true`, or use LangSmith traces (`LANGSMITH_TRACING_V2=true`).
