from operator import itemgetter

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

import config

from .data_processing import load_and_chunk_documents
from .llm_factory import get_embedding_model, get_llm
from .prompt_manager import PromptManager


def create_sentence_level_retrievers(docs):


def create_sentence_level_retrievers(docs):
    """
    Создает и возвращает два ретривера, работающих на уровне предложений.
    """
    # 1. Разбиваем документы на предложения для точного поиска
    from langchain_text_splitters import SentenceTransformersTokenTextSplitter

    sentence_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=256
    )
    sentences = sentence_splitter.split_documents(docs)

    # 2. Создаем временную векторную базу ПРЕДЛОЖЕНИЙ
    sentence_embeddings = get_embedding_model()
    sentence_vector_store = Chroma.from_documents(sentences, sentence_embeddings)

    # 3. Создаем два ретривера на основе предложений
    semantic_retriever = sentence_vector_store.as_retriever(search_kwargs={"k": 20})
    keyword_retriever = BM25Retriever.from_documents(sentences)
    keyword_retriever.k = 20

    return semantic_retriever, keyword_retriever


def get_windowed_context(docs):
    """
    Принимает найденные предложения и "собирает" вокруг них контекст.
    (В реальном проекте здесь была бы более сложная логика поиска по исходному тексту,
    но для нашей цели мы просто объединим найденные предложения,
    так как они уже содержат достаточно информации).
    """
    return "\n\n".join([doc.page_content for doc in docs])


def create_ultimate_chain():
    """
    Создает ультимативную RAG-цепочку:
    1. Гибридный поиск (BM25 + семантика) на уровне предложений.
    2. "Оконное" расширение контекста.
    3. Финальный ре-ранкинг.
    """
    print("Создание ультимативной RAG-цепочки...")

    # --- 1. Подготовка документов и ретриверов ---
    # Мы загружаем и чанким документы один раз
    all_chunks = load_and_chunk_documents(config.SOURCE_DOCS_PATH)

    # Создаем ретриверы, работающие на уровне предложений
    semantic_sent_retriever, keyword_sent_retriever = create_sentence_level_retrievers(
        all_chunks
    )

    # --- 2. Собираем ансамблевый ретривер для поиска по предложениям ---
    print("Создание ансамблевого ретривера для поиска по предложениям...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_sent_retriever, keyword_sent_retriever],
        weights=[0.5, 0.5],  # Даем равный вес смыслу и ключевым словам
    )

    # --- 3. Добавляем ре-ранкер ---
    print("Инициализация ре-ранкера (FlashRank)...")
    flashrank_client = Ranker(
        model_name=config.RERANKING_MODEL, cache_dir=config.CACHE_DIR
    )
    compressor = FlashrankRerank(
        client=flashrank_client, top_n=7
    )  # Берем чуть больше контекста

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # --- 4. Финальная цепочка для генерации ответа ---
    final_llm = get_llm()
    prompt_manager = PromptManager()

    def render_prompt(inputs):
        """Рендерит промпт через PromptManager и возвращает список сообщений."""
        text = prompt_manager.render("ultimate_chain", **inputs)
        return [HumanMessage(content=text)]

    # Собираем все в единый конвейер
    ultimate_chain = (
        {
            "context": itemgetter("question") | final_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(
            context=(lambda x: get_windowed_context(x["context"]))
        )
        | RunnableLambda(render_prompt)
        | final_llm
        | StrOutputParser()
    )

    print("Ультимативная RAG-цепочка успешно создана.")
    return ultimate_chain
