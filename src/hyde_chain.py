from operator import itemgetter

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from . import config
from .data_processing import load_and_chunk_documents
from .vector_store import load_vector_store


def format_docs(docs):
    """Форматирует найденные документы в единую строку."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_hyde_rag_chain():
    """
    Создает и возвращает RAG-цепочку, улучшенную с помощью HyDE.
    """
    # 1. Настройка моделей
    final_llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)
    hyde_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 2. Цепочка для генерации гипотетического ответа (СОВРЕМЕННЫЙ СИНТАКСИС)
    hyde_prompt = ChatPromptTemplate.from_template(
        "Напиши короткий, гипотетический ответ на следующий вопрос, как если "
        "бы он был взят из официального документа по охране труда: {question}"
    )
    # Старый LLMChain заменен на стандартную и надежную LCEL-цепочку.
    # Она гарантированно возвращает СТРОКУ, а не словарь.
    hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()

    # 3. Настройка ретривера
    docs = load_and_chunk_documents(config.SOURCE_DOCS_PATH)
    vector_store = load_vector_store(docs)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    flashrank_client = Ranker(
        model_name=config.RERANKING_MODEL, cache_dir=config.CACHE_DIR
    )
    compressor = FlashrankRerank(client=flashrank_client, top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 4. Финальный промпт
    final_template = """Ты — ИИ-ассистент. Используя ТОЛЬКО предоставленный 
    Контекст, дайте четкий и исчерпывающий ответ на Вопрос. Не придумывайте информацию. 
    Отвечай на русском языке.

    Контекст:
    {context}

    Вопрос:
    {question}

    Ответ:"""

    final_prompt = ChatPromptTemplate.from_template(final_template)

    # 5. Сборка основной цепочки
    hyde_rag_chain = (
        {"question": RunnablePassthrough()}  # Принимаем исходный вопрос
        | RunnablePassthrough.assign(
            # Передаем вопрос в hyde_chain, чтобы получить гипотетический ответ (строку),
            # а затем ищем по нему контекст.
            context=itemgetter("question")
            | hyde_chain
            | compression_retriever
        )
        | RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | final_prompt
        | final_llm
        | StrOutputParser()
    )

    return hyde_rag_chain
