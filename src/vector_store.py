# src/vector_store.py

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
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
        persist_directory=config.CHROMA_DB_PATH
    )
    
    print("Векторная база данных успешно создана и сохранена.")
    return vector_store

def load_vector_store() -> Chroma:
    """
    Загружает существующее векторное хранилище с диска.

    :return: Объект базы данных Chroma.
    """
    embeddings_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma(
        persist_directory=config.CHROMA_DB_PATH,
        embedding_function=embeddings_model
    )
    
    print("Векторная база данных успешно загружена с диска.")
    return vector_store
