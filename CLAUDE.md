# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Safety Compliance Assistant** is a RAG system for analyzing workplace safety regulations (Russian ГОСТ, СНиП, СП standards). It uses hybrid retrieval (semantic + BM25), FlashRank reranking, and a multi-agent LangGraph workflow for quality control.

**Key Technologies**: Python 3.11+, LangChain, LangGraph, ChromaDB, Docling, FlashRank, Streamlit

## Development Commands

### Core Operations
```bash
# Setup virtual environment (first time only)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Index documents (converts PDF/DOCX to MD, creates vector embeddings)
python index.py

# Launch Streamlit web interface
streamlit run app.py

# Run full evaluation suite (requires LANGSMITH_API_KEY)
python eval/run_full_evaluation.py
```

### Testing
```bash
# Activate venv first (if not already activated)
source venv/bin/activate

# Run all tests with verbose output
pytest -v

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests

# Run single test file
pytest tests/test_retrieval_metrics.py -v
```

### Linting and Formatting
```bash
# Format code with Black
black .

# Lint with Ruff
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

### Evaluation and Metrics
```bash
# Run A/B testing between configurations
python run_ab_test.py

# Check if metrics meet target thresholds
python scripts/check_target_metrics.py

# Compare current run against baseline
python scripts/compare_with_baseline.py

# Analyze metric trends over time
python scripts/analyze_trends.py
```

### Dataset Management
```bash
# Generate new test questions from documents
python scripts/generate_questions.py

# Add questions to evaluation dataset
python scripts/add_questions_to_dataset.py

# Parse Perplexity-generated datasets
python scripts/parse_perplexity_dataset.py
```

### Prompt Management
```bash
# Validate all prompts in registry
python scripts/validate_prompts.py

# Test rendering of specific prompt versions
# (Set env vars like PROMPT_RESEARCH_AGENT_VERSION=v2)
DEBUG_PROMPTS=true python -c "from src.prompt_manager import PromptManager; pm = PromptManager(); print(pm.render('research_agent', question='test', context='test'))"
```

## Architecture

### Multi-Agent Workflow (LangGraph)

The system orchestrates three specialized agents in a graph-based workflow:

1. **RelevanceChecker** (`agents/relevance_checker.py`): Classifies user questions
   - `CAN_ANSWER`: Question is about workplace safety
   - `PARTIAL`: Partially relevant, needs clarification
   - `CANNOT_ANSWER`: Off-topic, spam, or chit-chat

2. **ResearchAgent** (`agents/research_agent.py`): Generates draft answers using retrieved context

3. **VerificationAgent** (`agents/verification_agent.py`): Validates answers against source documents
   - Checks for hallucinations
   - Verifies factual accuracy
   - Sends feedback for revision if needed (max 2 loops)

**Workflow Implementation**: `agents/workflow.py` defines the LangGraph state machine with conditional edges and retry logic.

### RAG Pipeline

**Entry Point**: `src/final_chain.py` implements the hybrid retrieval chain:

1. **Semantic Search**: ChromaDB vector search (top-K documents)
2. **Keyword Search**: BM25Retriever for lexical matching
3. **Ensemble**: Combines results with configurable weights (default: [0.6, 0.4])
4. **Reranking**: FlashRank reranks top candidates (final top-5)
5. **Generation**: LLM generates answer from reranked context

**Alternative Chain**: `src/ultimate_chain.py` provides a simpler non-agentic chain for comparison.

### Prompt Management System

**All prompts are version-controlled templates** using Jinja2:

- **Registry**: `prompts/registry.yaml` maps prompt IDs to versioned template files
- **Templates**: `prompts/agents/*.j2` and `prompts/chains/*.j2`
- **Manager**: `src/prompt_manager.py` handles rendering with variable substitution
- **Version Control**: Override active version via environment variables:
  ```bash
  PROMPT_RESEARCH_AGENT_VERSION=v2 streamlit run app.py
  ```

**When editing prompts**: Always update both the template file and `registry.yaml` if adding new versions. Use `scripts/validate_prompts.py` to verify integrity.

### LLM Abstraction

`src/llm_factory.py` provides unified interface for multiple LLM providers:
- **GigaChat** (default for Russian regulatory text)
- **OpenAI** (GPT-4o-mini, GPT-4o)

Switch providers via `.env`:
```bash
LLM_PROVIDER=gigachat  # or openai
MODEL_NAME=GigaChat    # or gpt-4o-mini
```

### Vector Store

`src/vector_store.py` manages ChromaDB:
- **Embeddings**: Supports OpenAI, HuggingFace API, or local models
- **Persistence**: Data stored in `CHROMA_DB_PATH` (default: `./chroma_db_gigachat`)
- **Collection**: Named via `CHROMA_COLLECTION_NAME` (default: `documents`)

### Document Processing

`src/file_handler.py` handles ingestion:
- **Supported Formats**: PDF, DOCX, Markdown
- **Converter**: Docling library for PDF/DOCX → Markdown conversion
- **Chunking**: RecursiveCharacterTextSplitter (default: 1200 chars, 150 overlap)

**Source Documents**: Place files in `source_docs/` directory before running `python index.py`.

## Configuration

**Environment Variables**: Copy `.env.example` → `.env` and configure:

**Required**:
- `GIGACHAT_CREDENTIALS` (or `OPENAI_API_KEY`)
- `OPENAI_API_KEY` (for embeddings, even if using GigaChat for LLM)

**Optional**:
- `LANGSMITH_API_KEY`: For evaluation and tracing
- `LANGSMITH_PROJECT`: Project name for LangSmith
- `WANDB_API_KEY`: For experiment tracking

**Tunable Parameters**:
- `CHUNK_SIZE`: Default 1200
- `CHUNK_OVERLAP`: Default 150
- `VECTOR_SEARCH_K`: Default 10
- `HYBRID_RETRIEVER_WEIGHTS`: Default [0.6, 0.4] (semantic, keyword)

**Settings Module**: `config/settings.py` loads and validates all configuration using pydantic-settings.

## Evaluation Framework

**Metrics** (target values in parentheses):
- **Correctness** (>7.0/10): Semantic similarity to ground truth
- **Faithfulness** (>0.85): No hallucinations vs. source context
- **Answer Relevance** (>0.80): Alignment with user question
- **P95 Latency** (<15s): Response time

**Evaluation Datasets**: `tests/dataset.csv` contains question/answer pairs

**Metric Implementations**:
- `src/retrieval_metrics.py`: Retrieval-specific metrics
- `src/advanced_generation_metrics.py`: Generation quality metrics
- `src/custom_evaluators.py`: Custom domain-specific evaluators

**CI/CD**: GitHub Actions workflow (`.github/workflows/evaluation.yml`) runs evaluation on commits.

## Key Files Reference

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface entry point |
| `index.py` | Document indexing script |
| `agents/workflow.py` | LangGraph multi-agent orchestration |
| `src/final_chain.py` | Hybrid RAG chain (production) |
| `src/ultimate_chain.py` | Simple RAG chain (baseline) |
| `src/prompt_manager.py` | Version-controlled prompt rendering |
| `src/llm_factory.py` | LLM provider abstraction |
| `src/vector_store.py` | ChromaDB vector store interface |
| `prompts/registry.yaml` | Prompt version registry |
| `config/settings.py` | Configuration management |
| `eval/run_full_evaluation.py` | Full evaluation suite |
| `tests/dataset.csv` | Evaluation question/answer dataset |

## Common Patterns

### Adding a New Agent

1. Create agent file in `agents/` (e.g., `agents/summary_agent.py`)
2. Create prompt template in `prompts/agents/` (e.g., `summary_v1.j2`)
3. Register prompt in `prompts/registry.yaml`:
   ```yaml
   summary_agent:
     active_version: "v1"
     versions:
       v1: "agents/summary_v1.j2"
   ```
4. Add node to workflow in `agents/workflow.py`:
   ```python
   wf.add_node("summarize", self._summarize_step)
   wf.add_edge("research", "summarize")
   ```

### Modifying Retrieval Strategy

Edit `src/final_chain.py`:
- Adjust `settings.VECTOR_SEARCH_K` for more/fewer candidates
- Modify `settings.HYBRID_RETRIEVER_WEIGHTS` to balance semantic vs keyword
- Change `top_n=5` in FlashrankRerank for different reranking cutoff

### Running Experiments

1. Modify configuration in `.env` or code
2. Run A/B test: `python run_ab_test.py`
3. Check metrics: `python scripts/check_target_metrics.py`
4. Compare to baseline: `python scripts/compare_with_baseline.py`
5. Log results to LangSmith for tracking

### Debugging Failed Tests

1. Check error logs in `analysis/error_reports/`
2. Run test with verbose output: `pytest tests/test_file.py -v -s`
3. Enable prompt debugging: `DEBUG_PROMPTS=true pytest tests/test_file.py`
4. Review LangSmith traces (if `LANGSMITH_TRACING_V2=true`)

## Important Notes

- **Prompt Changes**: Always validate with `scripts/validate_prompts.py` before committing
- **Russian Language**: System is optimized for Russian regulatory documents
- **GigaChat Default**: GigaChat performs better on Russian legal text than GPT-4
- **Reindexing**: Run `python index.py` after adding documents to `source_docs/`
- **FlashRank Performance**: CPU-intensive; expect 2-3s latency on reranking
- **Max Reruns**: Verification agent allows 2 revision loops before returning final answer
- **Test Data**: Keep `tests/dataset.csv` updated when adding new domain knowledge
