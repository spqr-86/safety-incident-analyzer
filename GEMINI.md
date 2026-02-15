# Project Overview

This project is an AI Safety Compliance Assistant, a RAG (Retrieval-Augmented Generation) system designed to analyze regulatory documents related to labor protection. It is built with Python and utilizes a multi-agent approach with LangGraph to ensure the quality of its responses.

The core technologies used are:
- **Python 3.11+**
- **Streamlit** for the user interface.
- **LangChain** and **LangGraph** for the LLM framework.
- **OpenAI (GPT-4o)** and **Google Gemini 3** as LLM providers.
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
pip install -r requirements.txt
```

**2. Configure the environment:**
Create a `.env` file in the project root (you can use `.env.example` as a template):
```env
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
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
- **Architecture:** The project follows a multi-agent RAG architecture orchestrated by LangGraph. The core logic is located in the `src/` and `agents/` directories. The configuration is managed in the `config/` directory. Detailed documentation about the architecture can be found in `docs/architecture/README.md`.
