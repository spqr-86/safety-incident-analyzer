import os
import sys
import streamlit as st

# --- Умный FIX для ChromaDB/SQLite3 в облаке ---
if os.path.exists("/home/adminuser/venv/bin/python"):
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------

import config
from src.chain import create_final_rag_chain # <-- Изменили импорт!

# --- Конфигурация страницы ---
st.set_page_config(page_title="AI Safety Compliance Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Safety Compliance Assistant")
st.caption(f"Ваш ИИ-помощник по нормативной документации. Модель: {config.MODEL_NAME}")

# --- Загрузка и кеширование ресурсов ---
@st.cache_resource
def load_resources():
    if not os.path.exists(config.CHROMA_DB_PATH) or not os.listdir(config.CHROMA_DB_PATH):
        st.error(f"База данных не найдена. Запустите 'python index.py' для ее создания.")
        return None, None
    try:
        chain, retriever = create_final_rag_chain()
        return chain, retriever
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке ресурсов: {e}")
        return None, None

# --- Основная логика приложения ---
rag_chain, retriever = load_resources()

if rag_chain and retriever:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Здравствуйте! Какой у вас вопрос по нормативной документации?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Задайте ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # --- ИСПОЛЬЗУЕМ НОВУЮ, НАДЕЖНУЮ ЛОГИКУ ---
            try:
                retrieved_docs = retriever.invoke(user_query)
            except Exception:
                retrieved_docs = []
            
            # Настоящий стриминг с помощью st.write_stream
            response = st.write_stream(
                rag_chain.stream({
                    "question": user_query,
                    "chat_history": st.session_state.get("messages", [])
                })
            )

            # Отображаем источники после ответа
            if retrieved_docs:
                with st.expander("Показать источники"):
                    for doc in retrieved_docs:
                        # ... (код для отображения источников остается таким же)
                        st.text(doc.page_content) # Упрощенный вывод
                        st.caption(f"Источник: {doc.metadata.get('source', 'N/A')}")
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Приложение не может быть запущено...")