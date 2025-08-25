# index.py
import os
import shutil

from dotenv import load_dotenv

from config.settings import settings
from src.file_handler import DocumentProcessor
from src.vector_store import create_vector_store
from utils.logging import logger

load_dotenv()


def _collect_paths(root_dir: str, allowed_exts: list[str]) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                paths.append(os.path.join(dirpath, name))
    return paths


def main():
    logger.info("Запуск процесса индексации...")

    if os.path.exists(settings.CHROMA_DB_PATH):
        logger.info(f"Удаление старой базы данных из: {settings.CHROMA_DB_PATH}...")
        shutil.rmtree(settings.CHROMA_DB_PATH, ignore_errors=True)

    # Собираем все файлы допустимых типов
    file_paths = _collect_paths(settings.SOURCE_DOCS_PATH, settings.ALLOWED_TYPES)
    if not file_paths:
        logger.warning(
            f"В папке {settings.SOURCE_DOCS_PATH} не найдено подходящих файлов."
        )
        return

    # Обработка через новый DocumentProcessor
    processor = DocumentProcessor()
    chunks = processor.process(file_paths)

    if chunks:
        logger.info(f"Всего будет проиндексировано {len(chunks)} чанков.")
        create_vector_store(chunks)  # твоя функция сохраняет в Chroma
        logger.info("Индексация успешно завершена.")
    else:
        logger.warning("Чанки не получены. Проверьте документы/конвертацию.")


if __name__ == "__main__":
    main()
