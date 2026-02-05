import re
from typing import List, Optional, Dict, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever


class ApplicabilityRetriever(BaseRetriever):
    vector_store: VectorStore
    bm25_retriever: BM25Retriever
    search_kwargs: Dict[str, Any] = {"k": 10}
    weights: List[float] = [0.6, 0.4]  # Semantic, Keyword

    def _extract_params(self, query: str) -> str:
        # Simple heuristic extraction for now
        # In a full implementation, this would use a fast NER or keyword matcher
        roles = r"(водитель|сварщик|электрик|слесарь|руководитель|мастер|директор|бухгалтер|оператор|работник)"
        works = r"(сварка|высот[ае]|погрузк|ремонт|осмотр|монтаж|очистк|уборк)"
        premises = r"(цех|склад|кабинет|офис|территори|помещени)"

        found_roles = re.findall(roles, query, re.IGNORECASE)
        found_works = re.findall(works, query, re.IGNORECASE)
        found_premises = re.findall(premises, query, re.IGNORECASE)

        parts = []
        if found_roles:
            parts.extend(found_roles)
        if found_works:
            parts.extend(found_works)
        if found_premises:
            parts.extend(found_premises)

        return " ".join(parts)

    def _get_applicability_query(self, query: str) -> str:
        extracted = self._extract_params(query)
        if not extracted:
            # Fallback to general norms
            return "общие требования охраны труда обязанности работника права и ответственность"
        else:
            # Boost specific params
            return f"требования безопасности {extracted}"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Topic Query (Semantic + Keyword)
        docs_topic_semantic = self.vector_store.similarity_search(
            query, **self.search_kwargs
        )
        docs_topic_keyword = self.bm25_retriever.invoke(query)

        # 2. Applicability Query (Semantic only usually enough, or reuse BM25)
        app_query = self._get_applicability_query(query)
        docs_app_semantic = self.vector_store.similarity_search(
            app_query, **self.search_kwargs
        )

        # Combine and deduplicate
        all_docs = docs_topic_semantic + docs_topic_keyword + docs_app_semantic
        unique_docs = {}
        for doc in all_docs:
            # Use page_content as key for deduplication (or source+page if available)
            # Assuming content uniqueness is good enough proxy
            key = doc.page_content[:100]  # Hash/key by prefix or full content
            if key not in unique_docs:
                unique_docs[key] = doc

        return list(unique_docs.values())
