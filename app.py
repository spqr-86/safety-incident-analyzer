import os
import sys
import re

import streamlit as st
from dotenv import load_dotenv

# --- Fix для ChromaDB/SQLite3 в некоторых окружениях (например, облако) ---
if os.path.exists("/home/adminuser/venv/bin/python"):
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# ---------------------------------------------------------------------------

# Локальные импорты
from config.settings import settings
from src.final_chain import create_final_hybrid_chain
from utils.logging import logger

# Multi-Agent RAG Workflow
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_classic.retrievers import EnsembleRetriever
    from agents.multiagent_rag import MultiAgentRAGWorkflow

    MAS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Multi-Agent RAG is not available: {e}")
    MAS_AVAILABLE = False

# Индексация «по кнопке»
try:
    import index as index_module  # index.py должен содержать main()

    INDEX_AVAILABLE = True
except Exception as e:
    logger.warning(f"Index module not importable: {e}")
    INDEX_AVAILABLE = False

load_dotenv()

# =========================
#     PAGE CONFIG & UI
# =========================
st.set_page_config(
    page_title="AI Safety Compliance Assistant", page_icon="🤖", layout="wide"
)

# Верхний заголовок
left, right = st.columns([0.8, 0.2], vertical_alignment="center")
with left:
    st.title("🤖 AI Safety Compliance Assistant")
    st.caption("Ваш ИИ-помощник по нормативной документации (СНиП, ГОСТ, ОТ, ПБ).")
with right:
    st.markdown(
        f"""
        <div style="text-align:right">
            <div style="display:inline-block;padding:6px 10px;border-radius:8px;
                        background:#eef2ff;border:1px solid #c7d2fe;font-size:12px;">
                LLM: <b>{settings.LLM_PROVIDER}</b> · <b>{settings.MODEL_NAME}</b>
            </div><br/>
            <div style="display:inline-block;margin-top:6px;padding:6px 10px;border-radius:8px;
                        background:#ecfeff;border:1px solid #a5f3fc;font-size:12px;">
                Embeddings: <b>{getattr(settings, "EMBEDDING_PROVIDER", "?")}</b>
                · <b>{getattr(settings, "EMBEDDING_MODEL_NAME", "?")}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
#        SIDEBAR
# =========================
with st.sidebar:
    st.subheader("⚙️ Режим работы")

    # MAS toggle
    if MAS_AVAILABLE:
        mas_mode = st.toggle("🧠 Multi-Agent RAG (Router → RAG → Verifier)", value=True)
    else:
        st.info("Multi-Agent RAG недоступен.")
        mas_mode = False

    st.divider()

    st.subheader("📚 Библиотека документов")
    st.caption(f"Путь к БД: `{settings.CHROMA_DB_PATH}`")

    # Кнопка пересоздания индекса
    if INDEX_AVAILABLE:
        if st.button("♻️ Переиндексировать библиотеку", use_container_width=True):
            with st.spinner("Индексация… это может занять несколько минут"):
                try:
                    index_module.main()
                    # Обновить кэш ресурсов после индексации
                    load_resources.clear()
                    st.success("Готово: библиотека переиндексирована.")
                except Exception as e:
                    st.error(f"Ошибка индексации: {e}")
    else:
        st.caption("Модуль индексации недоступен.")

    st.divider()

    st.subheader("🔧 Параметры отображения")
    show_sources_n = st.slider("Сколько источников показать", 3, 20, 8, 1)

    st.divider()
    if st.button("🧹 Очистить чат", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("last_answer", None)
        st.rerun()


# =========================
#     RESOURCE LOADING
# =========================
@st.cache_resource(show_spinner=False)
def load_resources():
    """
    Грузим один раз:
      - гибридную RAG-цепочку (retriever + LLM)
      - Multi-Agent RAG Workflow
    """
    # Проверяем наличие БД
    if not os.path.exists(settings.CHROMA_DB_PATH) or not os.listdir(
        settings.CHROMA_DB_PATH
    ):
        st.error("База данных не найдена. Запустите 'python index.py' для её создания.")
        return None, None, None

    try:
        chain, retriever, agent_retriever = create_final_hybrid_chain()
    except Exception as e:
        st.error(f"Произошла ошибка при подготовке RAG-цепочки: {e}")
        return None, None, None

    # Multi-Agent RAG Workflow (использует OpenAI по умолчанию)
    agent = None
    if MAS_AVAILABLE:
        try:
            agent = MultiAgentRAGWorkflow(agent_retriever, llm_provider="gemini")
        except Exception as e:
            logger.warning(f"Failed to init MultiAgentRAGWorkflow: {e}")

    return (chain, retriever, agent)


loaded = load_resources()
if not loaded or loaded[0] is None:
    st.warning("Приложение не может быть запущено…")
    st.stop()

rag_chain, hybrid_retriever, agent = loaded

if mas_mode and agent is None:
    st.sidebar.warning("Multi-Agent RAG не удалось инициализировать. Используется Classic RAG.")

# =========================
#     CHAT HISTORY INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Здравствуйте! Я использую Multi-Agent RAG с маршрутизацией и верификацией. "
            "Задайте вопрос по охране труда, и я найду ответ в нормативных документах.",
        }
    ]

# Рендерим историю
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Рендеринг картинок в истории сложнее, так как st.image создает отдельный блок
        # Проверим, есть ли картинки в контенте сообщения (если это сохраненный ответ)
        if m["role"] == "assistant":
            img_matches = re.findall(
                r"(static/visuals/proof_[a-f0-9]+\.png)", m["content"]
            )
            for img_path in img_matches:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Визуальное доказательство", width=600)

# =========================
#       CHAT INPUT
# =========================
user_query = st.chat_input(
    "Спросите, например: «Какие требования к ширине эвакуационных путей?»"
)
if user_query:
    # Показываем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Ветка ответа
    with st.chat_message("assistant"):
        if mas_mode and agent:
            # --- MULTI-AGENT RAG MODE ---
            with st.spinner("Multi-Agent RAG (Router → Search → Generate → Verify)..."):
                try:
                    result = agent.invoke(user_query)

                    answer = result.get("final_answer", "").strip()

                    if not answer:
                        answer = "Не удалось сформировать ответ."

                    st.markdown(answer)

                    # Проверка на наличие изображений в тексте ответа и их отрисовка
                    img_matches = re.findall(
                        r"(static/visuals/proof_[a-f0-9]+\.png)", answer
                    )
                    for img_path in img_matches:
                        if os.path.exists(img_path):
                            st.image(
                                img_path, caption="Визуальное доказательство", width=600
                            )

                    # Сохраним последний ответ для истории
                    st.session_state.last_answer = answer

                except Exception as e:
                    st.error(f"Ошибка агента: {e}")
                    answer = f"Извините, произошла ошибка: {e}"
                    logger.error(f"Agent error: {e}", exc_info=True)
        else:
            # --- RAG MODE (Legacy) ---
            try:
                with st.spinner("Готовлю ответ (Classic RAG)..."):
                    response_text = st.write_stream(
                        rag_chain.stream(
                            {
                                "question": user_query,
                                "chat_history": st.session_state.get("messages", []),
                            }
                        )
                    )
                st.session_state.last_answer = response_text
                answer = response_text
            except Exception as e:
                st.error(f"Ошибка RAG-цепочки: {e}")
                answer = f"Извините, произошла ошибка: {e}"

        # Источники (общие для обоих режимов)
        # Агент сам ищет, но мы можем показать топ документов из RAG для справки
        try:
            retrieved_docs = hybrid_retriever.invoke(user_query)[:show_sources_n]
        except Exception:
            retrieved_docs = []

        if retrieved_docs:
            with st.expander(
                f"🔎 Показать источники (топ-{len(retrieved_docs)})", expanded=False
            ):
                for i, doc in enumerate(retrieved_docs, start=1):
                    src = (
                        doc.metadata.get("source")
                        or doc.metadata.get("file_path")
                        or "N/A"
                    )
                    section = (
                        doc.metadata.get("section") or doc.metadata.get("header") or ""
                    )
                    preview = doc.page_content[:500].strip().replace("\n", " ")
                    st.markdown(
                        f"**{i}. Источник:** `{src}`  "
                        + (f" · *{section}*" if section else "")
                    )
                    st.code(preview, language="markdown")
                    st.divider()
        else:
            st.caption("Не удалось получить источники для отображения.")

    # Сохраняем сообщение ассистента в историю
    st.session_state.messages.append({"role": "assistant", "content": answer})
