
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.multiagent_rag import MultiAgentRAGWorkflow
from src.vector_store import load_vector_store
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
QUERY = "для кого проводится повторный инструктаж?"
LLM_PROVIDER = "gemini"
EMBEDDING_PROVIDER = settings.EMBEDDING_PROVIDER
# ---

async def main():
    """Initializes and runs the RAG workflow, printing timing for each event."""
    logger.info("Инициализация Chroma Vector Store...")
    try:
        vs = load_vector_store()
        retriever = vs.as_retriever()
        logger.info("Vector Store готов.")
    except Exception as e:
        logger.error(f"Не удалось инициализировать Vector Store: {e}")
        return

    logger.info(f"Инициализация RAG воркфлоу с провайдером: {LLM_PROVIDER}...")
    workflow = MultiAgentRAGWorkflow(retriever=retriever, llm_provider=LLM_PROVIDER)
    logger.info("Воркфлоу готов.")

    logger.info(f"--- ЗАПУСК ПРОЦЕССА ДЛЯ ВОПРОСА: '{QUERY}' ---")
    start_time = time.time()

    try:
        for event in workflow.stream_events(QUERY):
            current_time = time.time()
            elapsed = current_time - start_time
            print(f"[{elapsed:.2f}s] Event: {event}")

    except Exception as e:
        logger.error(f"Произошла ошибка во время выполнения: {e}", exc_info=True)

    finally:
        total_time = time.time() - start_time
        logger.info(f"--- ПРОЦЕСС ЗАВЕРШЕН ЗА {total_time:.2f}s ---")


if __name__ == "__main__":
    # On Windows, we need to set a different event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # In Python 3.7+, asyncio.run is available and preferred.
    # For simplicity and compatibility, we'll just run the async main function.
    # This might require `await main()` if running in an already async environment.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # This can happen if an event loop is already running (e.g., in Jupyter)
        # In that case, we just await it.
        if "cannot run loop while another loop is running" in str(e):
            # This is a bit of a hack for script execution.
            # A more robust solution would be needed for library use.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise

