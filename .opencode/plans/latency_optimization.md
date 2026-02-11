# Latency Optimization (Feb 2026)

## Goals
Reduce system latency from ~60s+ to <15s for simple queries, and improve UX.

## Changes Implemented

### 1. UX & Perceived Latency
- **Streaming Response:** Implemented simulated token streaming (typewriter effect) in `app.py` so users see progress immediately.
- **Status Updates:** Agent streams detailed status ("Searching...", "Thinking...") to the UI.
- **Visual Proofs:** Fixed logic to ensure screenshots are always displayed if found by the agent.

### 2. Search Optimization (Critical)
- **Problem:** `FlashRank` reranking (L-12 model) took ~35s on CPU for 20 documents.
- **Solution:** Disabled Reranking for the Agent's search tool (`search_documents`).
- **Compensation:** Increased retrieval `k` from 5 to 10 (returning 20 docs total) to maintain recall. The Agent (Gemini Flash) can handle the larger context window easily.
- **Result:** Search latency dropped from ~35s to ~4s.

### 3. Startup Optimization
- **BM25 Caching:** Added pickling for the BM25 index in `src/final_chain.py`.
- **Result:** Startup time reduced significantly after the first run.

### 4. Smart Routing & Verification
- **Router Agent:** Replaced Regex Filter with a Gemini Flash-based `RouterAgent` for better classification.
- **Smart Verification:** Implemented logic to SKIP the Verifier step if the Agent finds a high-confidence answer (Similarity > 0.85).
- **Result:** Simple queries are answered faster by skipping the extra LLM call.

## Status
- [x] Phase 1: UX & Quick Wins (Smart Verifier)
- [x] Phase 2: Router & Parallel Search (Router Agent implemented)
- [x] Phase 3: Infrastructure (BM25 Cache, Search Optimization)

## Next Steps
- [ ] Implement Semantic Cache (Redis/Chroma) for sub-second responses to repeated queries.
- [ ] Pre-compute "broken table" flags during indexing.
