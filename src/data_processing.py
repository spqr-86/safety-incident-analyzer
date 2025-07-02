from typing import List

from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_and_chunk_documents(directory_path: str) -> List[Document]:
    """
    Загружает все PDF-документы из указанной директории и разбивает их на чанки.

    :param directory_path: Путь к директории с PDF-файлами.
    :return: Список объектов Document (чанков).
    """
    print(f"Загрузка документов из директории: {directory_path}...")

    # Проверяем, существует ли директория
    import os

    if not os.path.isdir(directory_path):
        print(f"Ошибка: Директория не найдена по пути: {directory_path}")
        return []

    try:
        # Используем PyPDFDirectoryLoader для загрузки всех PDF из папки
        loader = PyPDFDirectoryLoader(directory_path)
        documents = loader.load()

        if not documents:
            print(
                f"Предупреждение: В директории {directory_path} не найдено PDF-файлов."
            )
            return []

        print(f"Загружено {len(documents)} страниц из всех документов.")

        # Создаем сплиттер для разбивки на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )

        # Разбиваем документы на чанки
        chunks = text_splitter.split_documents(documents)
        print(f"Все документы разбиты на {len(chunks)} чанков.")
        return chunks

    except Exception as e:
        print(f"Произошла ошибка при обработке документов: {e}")
        return []
