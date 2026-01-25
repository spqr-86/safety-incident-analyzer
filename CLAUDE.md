# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Safety Compliance Assistant — a RAG system for analyzing Russian occupational safety documentation (SNiP, GOST, SP, internal regulations). Uses hybrid retrieval (vector + BM25), FlashRank reranking, and a multi-agent quality control workflow built with LangGraph.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Index documents (required before first run)
python index.py

# Run Streamlit app
streamlit run app.py

# Run tests
pytest                           # all tests
pytest tests/test_retrieval_metrics.py  # single file
pytest -m unit                   # only unit tests
pytest -m "not slow"             # skip slow tests

# Linting (configured in pyproject.toml)
ruff check .
black .

# Evaluation
python eval/run_full_evaluation.py --limit 5   # quick eval
python eval/run_full_evaluation.py             # full eval
python scripts/compare_with_baseline.py        # compare with baseline
python run_ab_test.py                          # A/B test with LangSmith
```

## Architecture

### Data Flow
1. **Ingestion** (`index.py`, `src/file_handler.py`): Docling converts PDF/DOCX → Markdown → chunks → embeddings → ChromaDB
2. **Retrieval** (`src/final_chain.py`): Query → EnsembleRetriever (Chroma vector + BM25) → FlashRank rerank → top-5 context
3. **Generation** (`src/final_chain.py`): Context + query → LLM → answer

### Multi-Agent Workflow (`agents/workflow.py`)
LangGraph state machine with 3 agents:
- **RelevanceChecker** (`agents/relevance_checker.py`): Gates whether documents can answer the question (CAN_ANSWER/PARTIAL/NO)
- **ResearchAgent** (`agents/research_agent.py`): Generates draft answer from context
- **VerificationAgent** (`agents/verification_agent.py`): Checks answer is supported by sources; can trigger re-research (max 2 loops)

### Key Modules
- `src/llm_factory.py`: Factory for LLM (GigaChat/OpenAI) and embeddings (OpenAI/HF/local)
- `src/vector_store.py`: ChromaDB operations with token-based batching
- `config/settings.py`: Pydantic settings from .env file

## Configuration

Copy `.env.example` to `.env`. Key settings:
- `LLM_PROVIDER`: `gigachat` | `openai`
- `EMBEDDING_PROVIDER`: `openai` | `hf_api` | `local`
- `GIGACHAT_CREDENTIALS` or `OPENAI_API_KEY` required for LLM
- `OPENAI_API_KEY` typically needed for embeddings

ChromaDB path auto-suffixes by provider (e.g., `chroma_db_gigachat`).

## Evaluation System

Dataset: `tests/dataset.csv` (question, ground_truth pairs)

Target metrics:
- Correctness > 7.0/10
- Faithfulness > 0.85
- Answer Relevance > 0.80
- P95 Latency < 15s

CI runs on push/PR to main/develop and weekly for drift detection. See `.github/workflows/evaluation.yml`.

## Testing

Test markers defined in `pyproject.toml`:
- `@pytest.mark.slow` — slow tests
- `@pytest.mark.integration` — integration tests
- `@pytest.mark.unit` — unit tests
