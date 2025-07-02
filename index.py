import os
import shutil

from dotenv import load_dotenv

import config
from src.data_processing import load_and_chunk_documents
from src.vector_store import create_vector_store

load_dotenv()


def main():
    """
    Основная функция для запуска процесса индексации.
    Сканирует папку с документами, обрабатывает их и создает векторную базу.
    """
    print("Запуск процесса индексации...")

    # Удаляем старую базу данных ТОЛЬКО для текущего провайдера
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Удаление старой базы данных из: {config.CHROMA_DB_PATH}...")
        shutil.rmtree(config.CHROMA_DB_PATH)

    all_chunks = load_and_chunk_documents(config.SOURCE_DOCS_PATH)

    # Если чанки были успешно созданы, создаем векторную базу
    if all_chunks:
        print(f"Всего будет проиндексировано {len(all_chunks)} чанков.")
        create_vector_store(all_chunks)
        print("Индексация успешно завершена.")
    else:
        print("Не удалось создать чанки для индексации. Проверьте ваши документы.")


if __name__ == "__main__":
    main()
