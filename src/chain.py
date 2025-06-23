# src/chain.py

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from .vector_store import load_vector_store
import config

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

CONDENSE_QUESTION_TEMPLATE = """
Учитывая историю предыдущего диалога и новый вопрос, переформулируй новый вопрос так, чтобы он был самостоятельным и полным. 
Он должен содержать весь необходимый контекст из истории диалога.
Это ОЧЕНЬ ВАЖНО: возвращай только сам переформулированный вопрос, без лишних слов и прелюдий.

История диалога:
{chat_history}

Новый вопрос: {question}

Самостоятельный вопрос:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
# ----------------------------------------------------


def create_conversational_chain():
    """
    Создает и возвращает RAG-цепочку с памятью, ре-ранкером и кастомной трансформацией вопроса.
    """
    vector_store = load_vector_store()
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    compressor = FlashrankRerank(top_n=3) # Указываем, что после ре-ранкинга нужно оставить топ-3
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever, 
        base_compressor=compressor
    )

    llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Собираем финальную цепочку
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
        chain_type="stuff",
        return_source_documents=True
    )
    
    return chain
