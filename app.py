# app.py

import streamlit as st
import os
import config  # Импортируем наш конфиг для доступа к путям
from dotenv import load_dotenv
from src.chain import create_rag_chain
from src.vector_store import load_vector_store

load_dotenv()

# --- Конфигурация страницы Streamlit ---
st.set_page_config(
    page_title="AI Safety Compliance Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Safety Compliance Assistant")
st.caption(f"Ваш ИИ-помощник по нормативной документации. Модель: {config.MODEL_NAME}")

# --- Кеширование ресурсов ---
# Эта функция будет выполнена только один раз, и ее результат сохранится в кеше.
# Это гарантирует, что мы не будем пересоздавать модель и базу при каждом действии пользователя.
@st.cache_resource
def load_resources():
    """
    Загружает RAG-цепочку и ретривер.
    Проверяет наличие векторной базы данных перед загрузкой.
    """
    # Проверка, существует ли база данных
    if not os.path.exists(config.CHROMA_DB_PATH) or not os.listdir(config.CHROMA_DB_PATH):
        st.error(f"База данных не найдена. Пожалуйста, запустите 'python index.py' в терминале для ее создания.")
        return None, None
    
    try:
        rag_chain = create_rag_chain()
        # Нам нужен ретривер отдельно, чтобы показать источники
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        return rag_chain, retriever
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке RAG-цепочки: {e}")
        return None, None

# --- Основная логика приложения ---
rag_chain, retriever = load_resources()

if rag_chain and retriever:
    # Поле для ввода вопроса
    user_query = st.text_input(
        "Задайте ваш вопрос:",
        placeholder="Например: Каковы требования к основной надписи на чертежах?"
    )

    if user_query:
        with st.spinner("Анализирую документы..."):
            # Получаем ответ от цепочки
            final_answer = rag_chain.invoke(user_query)
            
            # Получаем источники от ретривера
            retrieved_docs = retriever.invoke(user_query)

            # Отображаем ответ
            st.markdown("### Ответ:")
            st.info(final_answer)

            # Отображаем источники в выпадающем списке
            with st.expander("Показать источники, использованные для ответа"):
                for i, doc in enumerate(retrieved_docs):
                    st.subheader(f"Источник #{i+1}")
                    st.write(f"**Файл:** {doc.metadata.get('source', 'N/A')}")
                    st.write(f"**Страница:** {doc.metadata.get('page', 0) + 1}") # +1 для человеческого восприятия
                    st.write("**Содержимое:**")
                    # Используем st.text для сохранения форматирования
                    st.text(doc.page_content)
else:
    st.warning("Приложение не может быть запущено. Пожалуйста, проверьте ошибки выше.")
