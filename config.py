# --- Настройки моделей OpenAI ---
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TEMPERATURE = 0.0

# --- Настройки для разбиения текста (чанкинга) ---
CHUNK_SIZE = 1000  # Размер чанка в символах
CHUNK_OVERLAP = 200  # Перекрытие между чанками в символах

# --- Настройки путей к файлам и папкам ---
CHROMA_DB_PATH = "chroma_db"  # Папка для сохранения векторной базы данных
SOURCE_DOCS_PATH = "data"  # Папка с исходными PDF-документами
