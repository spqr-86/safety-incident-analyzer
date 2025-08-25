from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import ALLOWED_TYPES, MAX_FILE_SIZE, MAX_TOTAL_SIZE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Параметры для LLM и векторного хранилища
    LLM_PROVIDER: str = "gigachat"  # варианты: huggingface, openai, groq
    MODEL_NAME: str = "gpt-4o-mini"  # для OpenAI
    TEMPERATURE: float = 0.0

    # Параметры для эмбеддингов
    EMBEDDING_PROVIDER: str = "openai"  # варианты: openai, hf_api, nomic
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    # Параметры для FlashRank
    RERANKING_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    FLASHRANK_CACHE_DIR: str = "/tmp/flashrank_cache"

    # Параметры для индексации и обработки документов
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list[str] = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list[float] = [0.6, 0.4]

    LOG_LEVEL: str = "INFO"

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150

    SOURCE_DOCS_PATH: str = "./source_docs"

    # динамически подставим суффикс по провайдеру,
    # если путь не задан через .env
    def model_post_init(self, __context) -> None:
        if self.CHROMA_DB_PATH == "./chroma_db":
            object.__setattr__(
                self, "CHROMA_DB_PATH", f"chroma_db_{self.LLM_PROVIDER.lower()}"
            )


settings = Settings()
