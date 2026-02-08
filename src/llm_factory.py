import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

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
            max_retries=3,
        )
    else:
        raise ValueError(
            f"Неизвестный провайдер LLM: {provider}. "
            "Доступные варианты: 'openai'"
        )

    # Добавляем автоматические повторы при сетевых ошибках (в т.ч. SSL)
    # return llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    return llm


def get_gemini_llm(
    model_name: str = None,
    temperature: float = 0.0,
    thinking_budget: int = None,
    response_mime_type: str = None,
):
    """
    Factory for Gemini LLM with model selection and thinking budget.

    Args:
        model_name: Model to use (default: GEMINI_FAST_MODEL from settings)
        temperature: Temperature for generation (default: 0.0)
        thinking_budget: Token budget for Gemini thinking mode (None = disabled)
        response_mime_type: Response format, e.g. "application/json" (None = text)

    Returns:
        ChatGoogleGenerativeAI instance
    """
    if ChatGoogleGenerativeAI is None:
        raise ImportError(
            "Google Gemini package (langchain-google-genai) not installed. "
            "Please install it or use another provider."
        )

    api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in .env or environment variables."
        )

    model = model_name or settings.GEMINI_FAST_MODEL

    kwargs = dict(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=2048,
        timeout=settings.REQUEST_TIMEOUT,
    )
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    if response_mime_type is not None:
        kwargs["response_mime_type"] = response_mime_type

    return ChatGoogleGenerativeAI(**kwargs)


def get_vision_llm():
    """
    Возвращает LLM с поддержкой Vision (зрения).
    Используется для анализа скриншотов документов.
    """
    return get_gemini_llm()


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
