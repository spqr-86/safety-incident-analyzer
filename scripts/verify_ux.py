"""
End-to-end smoke test: удаляет семантический кэш и прогоняет вопрос
через MultiAgentRAGWorkflow с реальными ChromaDB, retriever и LLM.

Использование:
    python scripts/verify_ux.py                          # вопрос по умолчанию
    python scripts/verify_ux.py "какой срок хранения нарядов-допусков?"
    python scripts/verify_ux.py --keep-cache "вопрос"    # не удалять кэш
"""

import argparse
import logging
import os
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
DEFAULT_QUERY = "для кого проводится повторный инструктаж?"
LLM_PROVIDER = "gemini"
CACHE_FILE = "semantic_cache.json"
# ---


def clear_cache():
    """Удаляет семантический кэш, чтобы ответ шёл через полный пайплайн."""
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        cache_path.unlink()
        logger.info(f"Кэш {CACHE_FILE} удалён.")
    else:
        logger.info(f"Кэш {CACHE_FILE} не найден, пропускаем.")


def main():
    """Initializes and runs the RAG workflow, printing timing for each event."""
    parser = argparse.ArgumentParser(description="E2E smoke test для RAG workflow")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="Вопрос для тестирования")
    parser.add_argument("--keep-cache", action="store_true", help="Не удалять семантический кэш")
    args = parser.parse_args()

    query = args.query

    # 1. Удаляем кэш
    if not args.keep_cache:
        clear_cache()

    # 2. Инициализация
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

    # 3. Запуск
    logger.info(f"--- ЗАПУСК: '{query}' ---")
    start_time = time.time()
    final_answer = None

    try:
        for event in workflow.stream_events(query):
            elapsed = time.time() - start_time
            event_type = event.get("type", "unknown")

            if event_type == "status":
                print(f"  [{elapsed:6.2f}s] 📡 {event.get('text', '')}")
            elif event_type == "final":
                final_answer = event.get("answer", "")
                chunks = event.get("chunks_found", [])
                images = event.get("image_paths", [])
                print(f"  [{elapsed:6.2f}s] ✅ Финальный ответ получен")
                print(f"           Чанков: {len(chunks)}, Изображений: {len(images)}")
            else:
                print(f"  [{elapsed:6.2f}s] ❓ {event}")

    except Exception as e:
        logger.error(f"Ошибка во время выполнения: {e}", exc_info=True)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Время: {total_time:.2f}s")
    print(f"{'='*60}")

    if final_answer:
        print(f"\n📋 ОТВЕТ:\n{final_answer}")
    else:
        print("\n⚠️  Финальный ответ не получен!")


if __name__ == "__main__":
    main()
