from operator import itemgetter

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

import config

from .data_processing import load_and_chunk_documents

from .vector_store import load_vector_store


def format_docs(docs):
    """Форматирует найденные документы в единую строку."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_final_hybrid_chain():
    """
    Создает продвинутую RAG-цепочку, используя гибридный поиск
    (семантический + ключевые слова) и ре-ранкинг.
    """
    print("Создание финальной гибридной RAG-цепочки...")

    # --- 1. Подготовка документов ---
    docs = load_and_chunk_documents(config.SOURCE_DOCS_PATH)
    vector_store = load_vector_store(docs)

    # --- 2. Создаем нашего первого "детектива": Семантический ретривер ---
    print("Инициализация семантического ретривера (Chroma)...")
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # --- 3. Создаем нашего второго "детектива": Ключевой ретривер (BM25) ---
    # Он очень быстрый и не требует API.
    print("Инициализация ключевого ретривера (BM25)...")
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 10

    # --- 4. Собираем "элитный отряд": Ансамблевый ретривер ---
    # EnsembleRetriever объединит результаты обоих.
    # Мы даем больший "вес" семантическому поиску, но BM25 будет его страховать.
    print("Создание ансамблевого ретривера...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.6, 0.4],  # 60% важности семантике, 40% - ключевым словам
    )

    # --- 5. Добавляем "эксперта-криминалиста": Ре-ранкер ---
    # Он возьмет смешанные результаты от "отряда" и выберет лучшие.
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
    final_llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)

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
    return final_chain
