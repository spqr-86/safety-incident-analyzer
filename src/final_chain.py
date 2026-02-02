from operator import itemgetter

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

from config.settings import settings

from .llm_factory import get_llm
from .vector_store import load_vector_store
from .prompt_manager import PromptManager


def format_docs(docs):


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_final_hybrid_chain():
    print("Создание финальной гибридной RAG-цепочки...")

    # 1) Векторная БД
    vector_store = load_vector_store()

    # 2) Семантический ретривер
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.VECTOR_SEARCH_K}
    )

    # 3) Ключевой ретривер (BM25) из всех чанков
    all_data = vector_store.get(include=["metadatas", "documents"])
    all_docs_as_objects = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    keyword_retriever = BM25Retriever.from_documents(all_docs_as_objects)
    keyword_retriever.k = settings.VECTOR_SEARCH_K

    # 4) Ансамбль
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=settings.HYBRID_RETRIEVER_WEIGHTS,
    )

    # 5) Реранкер FlashRank
    flashrank_client = Ranker(
        model_name=settings.RERANKING_MODEL,
        cache_dir=getattr(settings, "FLASHRANK_CACHE_DIR", None),
    )
    compressor = FlashrankRerank(client=flashrank_client, top_n=5)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # 6) Генерация
    final_llm = get_llm()
    prompt_manager = PromptManager()

    def render_prompt(inputs):
        """Рендерит промпт через PromptManager и возвращает список сообщений."""
        text = prompt_manager.render("final_chain", **inputs)
        return [HumanMessage(content=text)]

    final_chain = (
        {
            "context": itemgetter("question") | final_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | RunnableLambda(render_prompt)
        | final_llm
        | StrOutputParser()
    )

    print("Финальная гибридная цепочка успешно создана.")
    return final_chain, final_retriever
