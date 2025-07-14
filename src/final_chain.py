from operator import itemgetter

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

import config

from .vector_store import load_vector_store

from .llm_factory import get_llm


def format_docs(docs):
    """Форматирует найденные документы в единую строку."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_final_hybrid_chain():
    """
    Создает продвинутую RAG-цепочку, используя гибридный поиск
    (семантический + ключевые слова) и ре-ранкинг.
    """
    print("Создание финальной гибридной RAG-цепочки...")

    # --- 1. Загрузка векторной базы ---
    vector_store = load_vector_store()

    # --- 2. Создаем cемантический ретривер ---
    print("Инициализация семантического ретривера (Chroma)...")
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # --- 3. Создаем ключевой ретривер (BM25) ---
    print("Извлечение документов из Chroma для BM25 ретривера...")
    # Метод .get() извлекает все записи из базы. Мы берем только сами документы.
    all_data = vector_store.get(include=["metadatas", "documents"])
    # Собираем полноценные объекты Document
    all_docs_as_objects = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    print("Инициализация ключевого ретривера (BM25) из существующих чанков...")
    keyword_retriever = BM25Retriever.from_documents(all_docs_as_objects)
    keyword_retriever.k = 10

    # --- 4. Собираем ансамблевый ретривер ---
    # EnsembleRetriever объединит результаты обоих.
    # Даем больший "вес" семантическому поиску.
    print("Создание ансамблевого ретривера...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.6, 0.4],  # 60% важности семантике, 40% - ключевым словам
    )

    # --- 5. Добавляем ре-ранкер ---
    print("Инициализация ре-ранкера (FlashRank)...")
    flashrank_client = Ranker(
        model_name=config.RERANKING_MODEL, cache_dir=config.CACHE_DIR
    )
    compressor = FlashrankRerank(client=flashrank_client, top_n=5)

    # Финальный ретривер с компрессией
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # --- 6. Финальная цепочка для генерации ответа ---
    final_llm = get_llm()

    final_template = """Вы — ИИ-ассистент.
    Используя ТОЛЬКО предоставленный Контекст, дайте четкий и исчерпывающий
    ответ на Вопрос. Не придумывайте информацию. Отвечай на русском языке.

    Контекст:
    {context}

    Вопрос:
    {question}

    Ответ:"""

    final_prompt = ChatPromptTemplate.from_template(final_template)

    final_chain = (
        {
            "context": itemgetter("question") | final_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | final_prompt
        | final_llm
        | StrOutputParser()
    )

    print("Финальная гибридная цепочка успешно создана.")
    return final_chain, final_retriever
