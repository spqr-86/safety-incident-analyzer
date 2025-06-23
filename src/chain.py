# src/chain.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser

from .vector_store import load_vector_store
import config

def create_rag_chain() -> Runnable:
    """
    Создает и возвращает полную RAG цепочку, готовую к использованию.
    
    :return: Объект RAG-цепочки.
    """
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)

    template = """
    Используй следующие фрагменты контекста, чтобы ответить на вопрос в конце.
    Если ты не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.
    Ответ должен быть максимально лаконичным и по делу. Предоставляй ответ только на русском языке.

    Контекст: {context}

    Вопрос: {question}

    Полезный ответ:"""
    
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain