# src/llm_factory.py

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_gigachat import GigaChat

# --- Важный импорт для локальных моделей ---
from langchain_community.embeddings import HuggingFaceEmbeddings
import config


def get_llm():
    """
    Фабричная функция для создания объекта LLM в зависимости от настроек в config.
    """
    provider = config.LLM_PROVIDER.lower()
    print(f"Инициализация LLM от провайдера: {provider}")

    if provider == "openai":
        return ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)

    elif provider == "gigachat":
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not credentials:
            raise ValueError(
                "Не найдены авторизационные данные для GigaChat. Проверьте ваш .env файл."
            )
        return GigaChat(
            credentials=credentials, verify_ssl_certs=False, scope="GIGACHAT_API_PERS"
        )

    else:
        raise ValueError(
            f"Неизвестный провайдер LLM: {provider}. Доступные варианты: 'openai', 'gigachat'"
        )


def get_embedding_model():
    """
    Фабричная функция для создания объекта модели эмбеддингов.
    Выбор модели теперь напрямую зависит от LLM_PROVIDER.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        print(f"Инициализация модели эмбеддингов от провайдера: {provider}")
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    elif provider == "gigachat":
        # --- Если LLM - GigaChat, используем русифицированный SBERT ---
        model_name = "ai-forever/sbert_large_nlu_ru"
        print(f"Инициализация локальной модели эмбеддингов: {model_name}")

        # Настройки для запуска на CPU
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}  # Важно для некоторых моделей

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    else:
        raise ValueError(f"Неизвестный провайдер LLM для эмбеддингов: {provider}.")
