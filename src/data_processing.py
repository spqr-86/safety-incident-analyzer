from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import config


def load_and_chunk_documents(doc_path: str) -> List[Document]:
    """
    Загружает документ из указанного пути и разбивает его на чанки.
    
    :param doc_path: Путь к PDF-документу.
    :return: Список объектов Document (чанков).
    """
    try:
        loader = PyPDFLoader(doc_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Документ {doc_path} успешно загружен и разбит на {len(chunks)} чанков.")
        return chunks
    except Exception as e:
        print(f"Ошибка при обработке документа {doc_path}: {e}")
        return []
    
