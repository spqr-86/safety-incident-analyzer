from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # <-- Импортируем родной LLM для LangChain

# Импорты LlamaIndex
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

import config


def create_ultimate_rag_chain():
    """
    Создает гибридную RAG-цепочку, объединяющую HyDE и Sentence Window Retriever.
    """
    print("Создание ультимативной RAG-цепочки (HyDE + Sentence Window)...")

    # --- 1. Настройка моделей ---
    # Модель для LlamaIndex (для внутренних операций)
    Settings.llm = LlamaOpenAI(model="gpt-4", temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Модель для HyDE (дешевая)
    hyde_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Создаем отдельную модель для финальной цепочки LangChain ---
    final_llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=config.TEMPERATURE)

    # ... (код для создания индекса и query_engine остается без изменений) ...
    print("Инициализация индекса Sentence Window...")
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, node_parser=node_parser)

    query_engine = index.as_query_engine(
        similarity_top_k=15,
        node_postprocessors=[
            # Здесь можно будет добавить ре-ранкер LlamaIndex
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    # --- 4. Создаем цепочку HyDE (без изменений) ---
    hyde_prompt = ChatPromptTemplate.from_template(
        """Пожалуйста, сгенерируй короткий абзац, который является гипотетическим, но реалистичным ответом на вопрос пользователя. Ответ должен быть написан в сухом, официальном стиле, характерном для нормативных актов и постановлений по охране труда.

Пример:
Вопрос: Какова периодичность повторного инструктажа?
Ответ: Повторный инструктаж по охране труда для работников проводится с периодичностью не реже одного раза в шесть календарных месяцев.

Теперь, по аналогии, ответь на следующий вопрос.
Вопрос: {question}"""
    )
    hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()

    # --- 5. Собираем финальную цепочку ---
    def retrieve_context(input_dict):
        question = input_dict["question"]
        print(f"Оригинальный вопрос: {question}")

        hypothetical_answer = hyde_chain.invoke({"question": question})
        print(f"Гипотетический ответ (HyDE): {hypothetical_answer}")

        retrieved_nodes = query_engine.retrieve(hypothetical_answer)
        retrieved_docs = [node.get_content() for node in retrieved_nodes]

        return "\n\n".join(retrieved_docs)

    final_prompt_template = """Вы — ИИ-ассистент. Используя ТОЛЬКО предоставленный Контекст, дайте четкий и исчерпывающий ответ на Вопрос. Не придумывайте информацию. Отвечай на русском языке.

Контекст:
{context}

Вопрос:
{question}

Ответ:"""
    final_prompt = ChatPromptTemplate.from_template(final_prompt_template)

    # Собираем финальную цепочку, используя LangChain-совместимую модель `final_llm`
    ultimate_chain = (
        {
            "context": retrieve_context,
            "question": itemgetter("question"),
        }
        | final_prompt
        | final_llm  # <-- Используем правильную, совместимую модель
        | StrOutputParser()
    )

    return ultimate_chain
