# Latency Optimization (Feb 2026)

## Goals
Reduce system latency from ~60s+ to <15s for simple queries, and improve UX.

## Phase 1: Completed ✅
### 1. UX & Perceived Latency
- **Streaming Response:** Implemented simulated token streaming (typewriter effect).
- **Status Updates:** Agent streams detailed status ("Searching...", "Thinking...") to the UI.
- **Visual Proofs:** Fixed logic to ensure screenshots are always displayed.

### 2. Search Optimization
- **Disabled Reranking:** Dropped search time from ~35s to ~4s by removing FlashRank (L-12) and increasing retrieval `k` to 10.
- **BM25 Caching:** Added pickling for the BM25 index to speed up startup.

### 3. Smart Routing & Verification
- **Router Agent:** Replaced Regex with LLM-based Router.
- **Smart Verification:** Implemented logic to SKIP Verifier if similarity > 0.85 (saving ~1.5s).

## Phase 2: Execution Optimization (Next Session) 🚀

### 1. Parallel Search (High Priority) ✅
- **Goal:** Reduce RAG Complex latency by 30-50% (currently sequential).
- **Task:** Implement `asyncio.gather` for multiple independent sub-queries in `rag_agent`.
- **Status:** Implemented `search_documents` and `visual_proof` tools in `src/agent_tools.py` (reverted to sync for compatibility, relying on LangGraph threading for parallel execution).

### 2. Speculative Execution (Medium Priority) ✅
- **Goal:** Hide Router latency (~1s).
- **Task:** Launch `Router` and `RAG Simple` in parallel.
- **Status:** Implemented in `agents/multiagent_rag.py` using `ThreadPoolExecutor` in `stream_events`. Agent prompt updated to use speculative chunks.

### 3. Semantic Cache (High Priority) ✅
- **Goal:** Instant (<0.1s) answers for repeated/similar questions.
- **Task:** Implement caching layer using `SentenceTransformer` embeddings.
- **Status:** Implemented `SemanticCache` in `src/semantic_cache.py` and integrated into `MultiAgentRAGWorkflow`.

### 4. Index-Time Pre-computation (Medium Priority)
- **Goal:** Save runtime tokens and analysis time.
- **Task:** Analyze chunks during indexing for broken tables/sentences.
- **Logic:** Flag `needs_visual_analyze=True` in metadata. Agent uses this flag directly without "thinking" or calling VLM unnecessarily.
