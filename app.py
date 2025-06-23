import os
import sys
import streamlit as st

# --- Умный FIX для ChromaDB/SQLite3 в облаке ---
if os.path.exists("/home/adminuser/venv/bin/python"):
    print("Обнаружено окружение Streamlit Cloud. Применяю фикс для SQLite3.")
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("Фикс для SQLite3 успешно применен.")
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать pysqlite3...")
# -----------------------------------------

import config
from src.chain import create_conversational_chain

# --- Конфигурация страницы ---
st.set_page_config(page_title="AI Safety Compliance Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Safety Compliance Assistant")
st.caption(f"Ваш ИИ-помощник по нормативной документации. Модель: {config.MODEL_NAME}")

# --- Загрузка и кеширование ресурсов ---
@st.cache_resource
def load_chain():
    if not os.path.exists(config.CHROMA_DB_PATH) or not os.listdir(config.CHROMA_DB_PATH):
        st.error(f"База данных не найдена. Пожалуйста, запустите 'python index.py' для ее создания.")
        return None
    try:
        chain = create_conversational_chain()
        return chain
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке RAG-цепочки: {e}")
        return None

# --- Основная логика приложения ---
rag_chain = load_chain()

if rag_chain:
    # Инициализация истории чата
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Здравствуйте! Какой у вас вопрос по нормативной документации?"}]

    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Обработка нового ввода пользователя
    if user_query := st.chat_input("Задайте ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Отображаем ответ ассистента, используя стриминг и плейсхолдеры
        with st.chat_message("assistant"):
            # Создаем плейсхолдеры, которые будем обновлять
            answer_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            full_response = ""
            source_documents = []

            # Итерируемся по потоку данных от цепочки
            for chunk in rag_chain.stream({"question": user_query, "chat_history": st.session_state.messages}):
                # Ловим и собираем кусочки ответа
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    answer_placeholder.markdown(full_response + "▌") # ▌ - эффект курсора

                # Ловим и сохраняем источники
                if "source_documents" in chunk:
                    source_documents = chunk["source_documents"]

            # Обновляем плейсхолдер ответа финальным текстом без курсора
            answer_placeholder.markdown(full_response)
            
            # Отображаем найденные источники в их плейсхолдере
            if source_documents:
                with sources_placeholder.expander("Показать источники"):
                    for i, doc in enumerate(source_documents):
                        st.subheader(f"Источник #{i+1}")
                        try:
                            source = doc.metadata.get('source', 'N/A').split('/')[-1]
                            page = doc.metadata.get('page', 0) + 1
                            st.write(f"**Файл:** {source}, **Страница:** {page}")
                        except Exception:
                            st.write(doc.metadata) # На случай, если метаданные другие
                        st.text(doc.page_content)
                        st.write("---")

        # Добавляем полный ответ ассистента в историю
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Приложение не может быть запущено. Убедитесь, что база данных создана, и проверьте ошибки выше.")