import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()  # Load .env file for GIGACHAT_CREDENTIALS

# --- Важный импорт для локальных моделей ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import settings


def get_llm():
    """
    Фабричная функция для создания
    объекта LLM в зависимости от настроек в config.
    """
    provider = settings.LLM_PROVIDER.lower()
    print(f"Инициализация LLM от провайдера: {provider}")

    if provider == "openai":
        llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            timeout=settings.REQUEST_TIMEOUT,
        )
    elif provider == "gigachat":
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not credentials:
            raise ValueError(
                "Не найдены авторизационные данные для GigaChat. "
                "Проверьте ваш .env файл."
            )
        llm = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            timeout=settings.REQUEST_TIMEOUT,
        )
    else:
        raise ValueError(
            f"Неизвестный провайдер LLM: {provider}. "
            "Доступные варианты: 'openai', 'gigachat'"
        )

    # Добавляем автоматические повторы при сетевых ошибках (в т.ч. SSL)
    return llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def get_embedding_model():
    provider = (settings.EMBEDDING_PROVIDER or "").lower()
    model = settings.EMBEDDING_MODEL_NAME

    # ✅ OpenAI
    if provider == "openai":
        return OpenAIEmbeddings(
            model=model or "text-embedding-3-small",
        )

    # ✅ HuggingFace Inference API (через huggingface_hub)
    if provider == "hf_api":
        client = InferenceClient(
            model=model or "intfloat/multilingual-e5-base",
        )

        # Оборачиваем в "легкий адаптер" для LangChain
        class HFEmbeddingsWrapper:
            def embed_query(self, text: str):
                return client.feature_extraction(text)

            def embed_documents(self, texts: list[str]):
                return [client.feature_extraction(t) for t in texts]

        return HFEmbeddingsWrapper()

    # ✅ Локальная модель (Sentence Transformers)
    if provider in {"local", "huggingface"}:
        return HuggingFaceEmbeddings(
            model_name=model or "ai-forever/sbert_large_nlu_ru",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER={provider}")
