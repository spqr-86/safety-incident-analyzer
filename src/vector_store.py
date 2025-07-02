from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma


from .llm_factory import get_embedding_model

import config


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Создает векторное хранилище из списка чанков и сохраняет его на диск.
    Эта функция должна вызываться только скриптом index.py.

    :param chunks: Список чанков для добавления в базу.
    :return: Объект базы данных Chroma.
    """
    print("Создание новой векторной базы данных...")
    embeddings_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=config.CHROMA_DB_PATH,
    )

    print(
        f"Векторная база данных успешно создана и сохранена в: {config.CHROMA_DB_PATH}"
    )
    return vector_store


def load_vector_store() -> Chroma:
    """
    Загружает существующее векторное хранилище с диска.

    :return: Объект базы данных Chroma.
    """
    print(f"Загрузка существующей векторной базы из: {config.CHROMA_DB_PATH}")
    embeddings_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=config.CHROMA_DB_PATH,
        embedding_function=embeddings_model,
    )
    return vector_store
