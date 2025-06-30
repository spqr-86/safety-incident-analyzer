from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from .vector_store import load_vector_store
import config

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from flashrank import Ranker
from operator import itemgetter

def format_docs(docs):
    """Форматирует найденные документы в единую строку."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_final_rag_chain():
    """
    Создает финальную RAG-цепочку с нуля с использованием LCEL для полного контроля.
    Эта версия исправляет ошибку TypeError.
    """

    # --- Загрузка базовых компонентов ---
    vector_store = load_vector_store()
    llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)

    # --- ШАГ 1: Создаем клиент FlashRank ---
    # Эта строка сама позаботится о скачивании модели, если ее нет.
    # Мы явно создаем объект Ranker, который будет выполнять всю работу.
    print("Инициализация клиента FlashRank...")
    flashrank_client = Ranker(model_name=config.RERANKING_MODEL, cache_dir=config.CACHE_DIR)
    print("Клиент FlashRank готов.")

    # --- Ретривер с ре-ранкером ---
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    compressor = FlashrankRerank(client=flashrank_client, top_n=2)
    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever, 
        base_compressor=compressor
    )

    # --- 1. Промпт для переформулировки вопроса ---
    condense_question_template = """Учитывая историю диалога и новый вопрос, переформулируй новый вопрос так, чтобы он был самостоятельным.
    
История диалога:
{chat_history}

Новый вопрос: {question}

Самостоятельный вопрос:"""
    condense_question_prompt = PromptTemplate.from_template(condense_question_template)

    # --- 2. Основной промпт для ответа ---
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты — вежливый и полезный AI-ассистент по нормативной документации. Отвечай на вопрос, основываясь ТОЛЬКО на следующем контексте. Если информации в контексте недостаточно, скажи, что не знаешь ответа.\n\nКонтекст:\n{context}"),
        # MessagesPlaceholder(variable_name="chat_history"), # Упрощаем, передавая историю в контекст
        ("human", "{question}")
    ])
    
    # --- 3. Функция для форматирования истории чата в строку ---
    def format_chat_history(chat_history):
        return "\n".join(f'{msg["role"]}: {msg["content"]}' for msg in chat_history)
    
    # --- 4. Собираем цепочку для генерации самостоятельного вопроса ---
    standalone_question_chain = (
        {
            "question": itemgetter("question"),
            "chat_history": lambda x: format_chat_history(x["chat_history"])
        }
        | condense_question_prompt
        | llm
        | StrOutputParser()
    )

    answer_chain = answer_prompt | llm | StrOutputParser()

    # --- 5. Собираем финальную цепочку ---
    final_chain = (
        {
            "context": RunnableBranch(
                (lambda x: x.get("chat_history"), standalone_question_chain), # ЕСЛИ история есть, ТО запустить переформулировку
                itemgetter("question")                                        # ИНАЧЕ просто взять вопрос
            ) | retriever | format_docs,
            "question": itemgetter("question")
        }
        | answer_chain 
    )

    return final_chain, retriever
