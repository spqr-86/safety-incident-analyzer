import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from operator import itemgetter

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from langchain_community.retrievers import LlamaIndexRetriever

# Укажем место для хранения индекса LlamaIndex
PERSIST_DIR = "./storage_llamaindex"

def get_llama_index():
    """
    Создает индекс LlamaIndex, если он не существует,
    или загружает его с диска.
    """
    if not os.path.exists(PERSIST_DIR):
        print(f"Индекс LlamaIndex не найден. Создаем новый в папке {PERSIST_DIR}...")
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # Сохраняем индекс на диск
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Индекс создан и сохранен.")
    else:
        print(f"Загрузка существующего индекса LlamaIndex из папки {PERSIST_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Индекс успешно загружен.")
    
    return index

def create_hybrid_rag_chain():
    """
    Создает и возвращает гибридную RAG-цепочку LangChain + LlamaIndex.
    """
    load_dotenv()
    
    # Получаем готовый индекс (созданный или загруженный)
    index = get_llama_index()
    
    # Остальная часть не меняется
    query_engine = index.as_query_engine(similarity_top_k=3)
    retriever = LlamaIndexRetriever(index=query_engine)

    template = """
      Вы — ИИ-ассистент для ответов на вопросы.
    Используйте только приведенные ниже фрагменты контекста для ответа на вопрос.
    Если вы не знаете ответа, просто скажите, что не знаете. Не пытайтесь выдумывать ответ.
    Отвечай на русском языке.

    Контекст: {context}

    Вопрос: {question}

    Полезный ответ:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4", temperature=0)

    hybrid_rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return hybrid_rag_chain