# Immediate Feedback UX Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide immediate visual feedback ("🚀 Starting...", "🔎 Found X docs") to the user during the 2-5s speculative execution delay.

**Architecture:**
- Inject `yield` statements in `stream_events` before blocking calls.
- Manually yield "Router" status based on speculative result (since the graph node will be skipped).

**Tech Stack:** Python, LangGraph.

### Task 1: Implement Immediate Status Updates

**Files:**
- Modify: `agents/multiagent_rag.py`

**Step 1: Yield Start Status**
Insert `yield {"type": "status", "text": "🚀 Анализирую запрос..."}` before `ThreadPoolExecutor`.

**Step 2: Yield Search Status**
After `future_search.result()` returns, check if chunks exist.
If yes, `yield {"type": "status", "text": f"🔎 Предварительно найдено {len(search_chunks)} документов..."}`.

**Step 3: Yield Router Status (Manual)**
Before the `self.compiled_workflow.stream` loop:
Check `initial_state["route_type"]`.
If it's `RAG_COMPLEX`, yield `{"type": "status", "text": "🧠 Вопрос сложный, анализирую детали..."}`.
If it's `RAG_SIMPLE`, yield `{"type": "status", "text": "🔎 Ищу ответ в документах..."}`.
(This replicates the logic that `_router_node` would have triggered if it ran).

**Step 4: Verify**
Run the app manually (since we can't easily unit test streaming timing without complex mocks).
Or write a simple script `scripts/verify_ux.py` that iterates `stream_events` and prints timing.

### Task 2: Cleanup

**Files:**
- None.

**Step 1: Commit**
`git add agents/multiagent_rag.py && git commit -m "feat(ux): immediate status updates during speculative execution"`
