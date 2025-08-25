LLM_PROVIDER = "gigachat"

# --- Настройки моделей OpenAI ---
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TEMPERATURE = 0.0

# --- Настройки для разбиения текста (чанкинга) ---
CHUNK_SIZE = 1000  # Размер чанка в символах
CHUNK_OVERLAP = 200  # Перекрытие между чанками в символах

# --- Настройки путей к файлам и папкам ---
CHROMA_DB_PATH = (
    f"chroma_db_{LLM_PROVIDER.lower()}"  # Папка для сохранения векторной базы данных
)
SOURCE_DOCS_PATH = "data"  # Папка с исходными PDF-документами


RERANKING_MODEL = "ms-marco-MiniLM-L-12-v2"
CACHE_DIR = "/tmp/flashrank_cache"
