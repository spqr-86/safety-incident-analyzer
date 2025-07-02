import os

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

    # Собираем пути ко всем PDF файлам в исходной папке
    pdf_files = [f for f in os.listdir(config.SOURCE_DOCS_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(
            f"Не найдены PDF файлы в папке '{config.SOURCE_DOCS_PATH}'. Индексация прервана."
        )
        return

    print(f"Найдено {len(pdf_files)} документов для индексации.")

    # Обрабатываем каждый документ и собираем все чанки в один список
    all_chunks = []
    for doc_name in pdf_files:
        doc_path = os.path.join(config.SOURCE_DOCS_PATH, doc_name)
        chunks = load_and_chunk_documents(doc_path)
        if chunks:
            all_chunks.extend(chunks)

    # Если чанки были успешно созданы, создаем векторную базу
    if all_chunks:
        print(f"Всего будет проиндексировано {len(all_chunks)} чанков.")
        create_vector_store(all_chunks)
        print("Индексация успешно завершена.")
    else:
        print("Не удалось создать чанки для индексации. Проверьте ваши документы.")


if __name__ == "__main__":
    main()
