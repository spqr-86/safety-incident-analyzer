from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

# --- Обновленные импорты из LlamaIndex ---
from llama_index.core import Settings  # <-- Импортируем Settings вместо ServiceContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# Явно импортируем модель для эмбеддингов
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI


def create_sentence_window_chain():
    """
    Создает RAG-цепочку с использованием нативной реализации
    Sentence Window Retriever в LlamaIndex.
    """
    load_dotenv()

    print("Создание продвинутой RAG-цепочки (Sentence Window LlamaIndex)...")

    # 1. Настройка глобальных параметров через Settings
    # Это новый, правильный способ.
    print("Настройка LLM и модели эмбеддингов...")
    Settings.llm = LlamaOpenAI(model="gpt-4", temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 2. Создаем "оконный" парсер (этот код не меняется)
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # 3. Загружаем документы и строим индекс.
    # Он автоматически подхватит наши глобальные Settings.
    print("Создание/загрузка индекса LlamaIndex...")
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, node_parser=node_parser)

    print("Индекс создан.")

    # 4. Создаем движок запросов (этот код не меняется)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    # 5. Интегрируем движок в LangChain Runnable (этот код не меняется)
    def query_engine_func(input_dict):
        return query_engine.query(input_dict["question"])

    sentence_window_rag_chain = (
        {"question": itemgetter("question")}
        | RunnableLambda(query_engine_func)
        | (lambda response: str(response))
    )

    return sentence_window_rag_chain
