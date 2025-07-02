from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Создает векторное хранилище из списка чанков и сохраняет его на диск.

    :param chunks: Список чанков для добавления в базу.
    :return: Объект базы данных Chroma.
    """
    embeddings_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=config.CHROMA_DB_PATH,
    )

    print("Векторная база данных успешно создана и сохранена.")
    return vector_store


def load_vector_store(docs):
    """
    Создает или загружает векторную базу данных Chroma из предоставленных документов.

    Args:
        docs: Список документов для индексации.

    Returns:
        Объект векторной базы данных Chroma.
    """
    print("Создание векторной базы данных Chroma из документов...")

    # Создаем эмбеддинги
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Создаем векторную базу из документов.
    # Она будет сохранена на диск в папку, указанную в DB_PATH.
    vector_store = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=config.CHROMA_DB_PATH
    )

    print("Векторная база данных успешно создана и сохранена.")
    return vector_store
