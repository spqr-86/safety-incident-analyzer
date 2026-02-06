from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Any

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocItem, SectionHeaderItem, TextItem, ListItem
from langchain.docstore.document import Document

from config import constants
from config.settings import settings
from utils.logging import logger

FileLike = Union[str, os.PathLike, io.BufferedIOBase, io.BytesIO, io.StringIO]

# ⚙️ Обновляем версию, так как формат хранения кардинально меняется
PIPELINE_VERSION = "v2.0-visual-coords"


@dataclass
class CacheEntry:
    timestamp: float
    chunks: List[Document]


class DocumentProcessor:
    """
    Обработчик файлов с поддержкой извлечения координат (BBox) для визуализации.
    Использует Docling для структурного парсинга.
    """

    def __init__(
        self,
        headers: Optional[
            List[Tuple[str, str]]
        ] = None,  # Deprecated, kept for interface compat
        chunk_size: Optional[int] = None,  # Deprecated
        chunk_overlap: Optional[int] = None,  # Deprecated
    ):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ленивая инициализация Docling
        self._docling = DocumentConverter()

    # ---------- публичные методы ----------

    def validate_files(self, files: Iterable[FileLike]) -> None:
        """Проверяет суммарный размер загружаемых файлов."""
        total = 0
        for f in files:
            size = self._safe_sizeof(f)
            if size is None:
                continue
            total += size

        if total and total > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit "
                f"({total // 1024 // 1024}MB provided)."
            )

    def process(self, files: Iterable[FileLike]) -> List[Document]:
        """Обработка файлов с кэшированием."""
        self.validate_files(files)

        all_chunks: List[Document] = []
        seen_chunk_hashes: set[str] = set()

        for file_obj in files:
            try:
                stream, display_name = self._get_stream_and_name(file_obj)

                # Хэш файла для кэша
                file_hash = self._hash_bytes_stream(stream)
                cache_path = self._cache_path_for(file_hash)

                if self._is_cache_valid(cache_path):
                    logger.info(f"[cache] {display_name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"[process] {display_name}")
                    stream.seek(0)
                    chunks = self._convert_and_extract(stream, display_name, file_hash)
                    self._save_to_cache(chunks, cache_path)

                # Дедупликация
                for ch in chunks:
                    # Уникальность определяем по тексту + координатам (если есть)
                    # Но для простоты пока по тексту, хотя разные bbox могут иметь один текст
                    content_hash = hashlib.sha256(
                        ch.page_content.encode("utf-8")
                    ).hexdigest()
                    if content_hash not in seen_chunk_hashes:
                        all_chunks.append(ch)
                        seen_chunk_hashes.add(content_hash)

            except Exception as e:
                logger.error(
                    f"Failed to process '{getattr(file_obj, 'name', str(file_obj))}': {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # ---------- конвертация и извлечение ----------

    def _convert_and_extract(
        self, stream: io.BufferedIOBase, source_name: str, file_hash: str
    ) -> List[Document]:
        """Конвертация через Docling и извлечение структурных чанков."""
        import tempfile

        # Docling требует файл на диске
        suffix = self._suffix_from_name(source_name)
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            stream.seek(0)
            tmp.write(stream.read())
            tmp.flush()

            # Конвертация
            try:
                res = self._docling.convert(tmp.name)
            except Exception as e:
                logger.error(f"Docling conversion failed for {source_name}: {e}")
                return []

            return self._process_docling_document(res.document, source_name)

    def _process_docling_document(self, doc: Any, source: str) -> List[Document]:
        """
        Итерация по структуре документа Docling.
        Сохраняем каждый элемент как Document с metadata.
        """
        chunks = []

        # Итерируемся по всем текстовым элементам (заголовки, параграфы, списки)
        # doc.texts() возвращает итератор по TextItem
        # В новой версии Docling структура может отличаться, используем безопасный подход

        # Попытка получить плоский список элементов, если поддерживается
        items = []
        if hasattr(doc, "texts"):
            items = list(doc.texts())
        elif hasattr(doc, "body") and hasattr(doc.body, "children"):
            # Fallback: рекурсивный обход, если doc.texts() недоступен
            items = self._flatten_items(doc.body.children)

        current_section = "Начало документа"

        for i, item in enumerate(items):
            text = item.text.strip()
            if not text:
                continue

            # Обновляем текущую секцию для контекста
            if isinstance(item, SectionHeaderItem):
                current_section = text

            # Метаданные
            meta = {
                "source": source,
                "type": self._get_item_type(item),
                "chunk_id": i,
                "section_context": current_section,
            }

            # Извлечение координат (Provenance)
            if hasattr(item, "prov") and item.prov:
                # Берем первое вхождение
                prov = item.prov[0]
                if hasattr(prov, "bbox") and prov.bbox:
                    # Сохраняем bbox как JSON-строку для совместимости с Chroma
                    # Формат Docling: [L, B, R, T] (обычно)
                    # Мы просто сохраняем как есть, VisualTool разберется
                    meta["bbox"] = json.dumps(
                        prov.bbox.as_tuple()
                        if hasattr(prov.bbox, "as_tuple")
                        else prov.bbox
                    )

                if hasattr(prov, "page_no"):
                    meta["page_no"] = prov.page_no

            # Создаем документ
            # Добавляем контекст секции в начало текста для лучшего поиска
            enriched_content = f"[{current_section}] {text}"

            chunks.append(Document(page_content=enriched_content, metadata=meta))

        return chunks

    def _flatten_items(self, children: List[Any]) -> List[Any]:
        """Рекурсивно собирает текстовые элементы."""
        result = []
        for child in children:
            if isinstance(child, (TextItem, ListItem, SectionHeaderItem)):
                result.append(child)
            if hasattr(child, "children"):
                result.extend(self._flatten_items(child.children))
        return result

    def _get_item_type(self, item: Any) -> str:
        if isinstance(item, SectionHeaderItem):
            return "header"
        if isinstance(item, ListItem):
            return "list_item"
        return "text"

    # ---------- кэш и утилиты (без изменений логики) ----------

    def _cache_path_for(self, file_hash: str) -> Path:
        key = hashlib.sha256(
            f"{file_hash}:{PIPELINE_VERSION}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.pkl"

    def _save_to_cache(self, chunks: List[Document], cache_path: Path) -> None:
        data = CacheEntry(timestamp=datetime.now().timestamp(), chunks=chunks)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        with open(cache_path, "rb") as f:
            data: CacheEntry = pickle.load(f)
        return data.chunks

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        max_age = timedelta(days=getattr(settings, "CACHE_EXPIRE_DAYS", 7))
        return cache_age < max_age

    def _safe_sizeof(self, f: FileLike) -> Optional[int]:
        try:
            if isinstance(f, (str, os.PathLike)):
                return Path(f).stat().st_size
            if hasattr(f, "seek") and hasattr(f, "tell"):
                cur = f.tell()
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(cur, os.SEEK_SET)
                return size
        except Exception:
            return None
        return None

    def _get_stream_and_name(self, f: FileLike) -> Tuple[io.BytesIO, str]:
        if isinstance(f, (str, os.PathLike)):
            p = Path(f)
            with open(p, "rb") as fh:
                data = fh.read()
            return io.BytesIO(data), p.name
        if hasattr(f, "read"):
            raw = f.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            return io.BytesIO(raw), getattr(f, "name", "uploaded_file")
        raise TypeError(f"Unsupported file type: {type(f)}")

    def _hash_bytes_stream(self, stream: io.BytesIO) -> str:
        stream.seek(0)
        h = hashlib.sha256()
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        stream.seek(0)
        return h.hexdigest()

    def _suffix_from_name(self, name: str) -> str:
        suf = Path(name).suffix.lower()
        return suf if suf else ".bin"
