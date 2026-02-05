from __future__ import annotations

import hashlib
import io
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from docling.document_converter import DocumentConverter
from langchain.docstore.document import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import constants
from config.settings import settings
from utils.logging import logger

FileLike = Union[str, os.PathLike, io.BufferedIOBase, io.BytesIO, io.StringIO]


# ⚙️ меняй это при изменении пайплайна (заголовки, сплиттеры и т.д.)
PIPELINE_VERSION = "v1.4-context-injection"


@dataclass
class CacheEntry:
    timestamp: float
    chunks: List[Document]


class DocumentProcessor:
    """
    Надёжный обработчик файлов:
      - кэш по контенту файла и версии пайплайна
      - поддержка path и file-like объектов
      - двухступенчатый сплиттинг (Markdown headers -> recursive)
      - дедупликация чанков
    """

    def __init__(
        self,
        headers: Optional[List[Tuple[str, str]]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.headers = headers or [
            ("#", "Section"),
            ("##", "SubSection"),
            ("###", "Paragraph"),
        ]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # берём из настроек, но можно переопределить аргументами
        self.chunk_size = chunk_size or getattr(settings, "CHUNK_SIZE", 1500)
        self.chunk_overlap = chunk_overlap or getattr(settings, "CHUNK_OVERLAP", 400)

        # ленивые инстансы
        self._docling = DocumentConverter()

        # подготовим сплиттеры
        self._md_splitter = MarkdownHeaderTextSplitter(self.headers)
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n### ",  # Новые пункты
                "\n\n",  # Абзацы
                "\n[а-я]\\)\\s",  # Подпункты а), б), в) - РЕГУЛЯРКА
                "\n",
                ". ",
                " ",
                "",
            ],
            is_separator_regex=True,
        )

    # ---------- публичные методы ----------

    def validate_files(self, files: Iterable[FileLike]) -> None:
        """Проверяет суммарный размер загружаемых файлов (где возможно)."""
        total = 0
        for f in files:
            size = self._safe_sizeof(f)
            if size is None:
                # не знаем размер (например, BytesIO) — пропускаем, но логируем
                logger.debug("File size not available, skipping in total size calc.")
                continue
            total += size

        if total and total > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit "
                f"({total // 1024 // 1024}MB provided)."
            )

    def process(self, files: Iterable[FileLike]) -> List[Document]:
        """Обработка файлов с кэшированием и дедупликацией чанков."""
        self.validate_files(files)

        all_chunks: List[Document] = []
        seen_chunk_hashes: set[str] = set()

        for file_obj in files:
            try:
                # 1) получить bytes-поток и нормализованное имя
                stream, display_name = self._get_stream_and_name(file_obj)

                # 2) рассчитать контент-хэш файла потоково
                file_hash = self._hash_bytes_stream(stream)
                cache_path = self._cache_path_for(file_hash)

                # 3) кэш или обработка
                if self._is_cache_valid(cache_path):
                    logger.info(f"[cache] {display_name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"[process] {display_name}")
                    # важно: вернуть курсор в начало перед чтением конвертером
                    stream.seek(0)
                    markdown = self._to_markdown(stream, display_name)
                    chunks = self._split_markdown(
                        markdown, source=display_name, file_hash=file_hash
                    )
                    self._save_to_cache(chunks, cache_path)

                # 4) дедупликация чанков
                for ch in chunks:
                    norm = self._normalize_text(ch.page_content)
                    ch_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                    if ch_hash not in seen_chunk_hashes:
                        all_chunks.append(ch)
                        seen_chunk_hashes.add(ch_hash)

            except Exception as e:
                logger.error(
                    f"Failed to process '{getattr(file_obj, 'name', str(file_obj))}': {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # ---------- конвертация и сплиттинг ----------

    def _to_markdown(self, stream: io.BufferedIOBase, display_name: str) -> str:
        """
        Конвертация в Markdown через Docling.
        Если понадобится — тут же можно добавить fallback на pypdf+ocr и т.п.
        """
        # Docling удобнее кормить как temp-файл. Сделаем NamedTemporaryFile.
        import tempfile

        suffix = self._suffix_from_name(display_name)
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            # скопируем поток в файл
            stream.seek(0)
            tmp.write(stream.read())
            tmp.flush()

            # конвертация
            doc = self._docling.convert(tmp.name)
            md = doc.document.export_to_markdown()

        if not md or not md.strip():
            raise ValueError(f"Empty markdown after conversion for '{display_name}'")

        # 1. Превращаем "46.\n" в "### Пункт 46\n"
        md = re.sub(r"^(\d+)\.\s*$", r"### Пункт \1", md, flags=re.MULTILINE)

        # 2. Исправляем "46. Текст" -> "### Пункт 46. Текст" (если это заголовок пункта)
        md = re.sub(r"^(\d+)\.\s+(?=[А-Я])", r"### Пункт \1. ", md, flags=re.MULTILINE)

        # 3. Дополнительно: ловим "53. Обучению подлежат:" и подобные конструкции
        md = re.sub(
            r"^(\d+)\.\s+([А-Яа-я\s]+:)$", r"### Пункт \1. \2", md, flags=re.MULTILINE
        )

        # 4. Header Healing: Промоушен римских разделов "IX. ТРЕБОВАНИЯ..." -> "## Раздел IX. ТРЕБОВАНИЯ..."
        # Это лечит ошибку, когда сплиттер пропускает разделы без решеток
        md = re.sub(
            r"^(?P<roman>I|V|X|L|C|D|M|[IVXLCDM]+)\.\s+(?P<title>[А-Я\s]{5,})$",
            r"## Раздел \g<roman>. \g<title>",
            md,
            flags=re.MULTILINE,
        )

        return md

    def _split_markdown(
        self, markdown: str, source: str, file_hash: str
    ) -> List[Document]:
        """Сначала режем по заголовкам, потом — рекурсивно на удобные куски с инъекцией контекста."""
        # 1. Режем на крупные куски по заголовкам
        md_sections = self._md_splitter.split_text(markdown)
        expanded: List[Document] = []

        for sec in md_sections:
            # 2. ФОРМИРУЕМ ПРЕФИКС КОНТЕКСТА (вынесено в отдельный метод)
            enriched_content, header_context = self._enrich_chunk_with_context(
                sec.page_content, source, sec.metadata
            )

            # 3. Рекурсивно режем уже обогащенный текст
            meta = dict(sec.metadata or {})
            meta.update(
                {
                    "source": source,
                    "file_hash": file_hash,
                    "pipeline_version": PIPELINE_VERSION,
                    "content_type": "markdown",
                    "header_context": header_context,  # Сохраняем и в метаданных тоже
                }
            )

            # Передаем список из одного Document с обогащенным текстом
            sub_docs = self._recursive_splitter.split_documents(
                [Document(page_content=enriched_content, metadata=meta)]
            )
            expanded.extend(sub_docs)

        # финальный проход: проставим порядковые номера
        for i, d in enumerate(expanded):
            d.metadata["chunk_id"] = i

        return expanded

    def _enrich_chunk_with_context(
        self, content: str, source: str, metadata: dict
    ) -> Tuple[str, str]:
        """
        Внедрение контекста (Context Injection):
        Берет заголовки из метаданных и пишет их первой строчкой в тексте чанка.
        """
        section = metadata.get("Section", "")
        subsection = metadata.get("SubSection", "")
        paragraph = metadata.get("Paragraph", "")

        # Собираем полный путь к текущему фрагменту для точной векторизации
        context_parts = []
        if section:
            context_parts.append(section)
        if subsection:
            context_parts.append(subsection)
        if paragraph:
            context_parts.append(paragraph)

        header_context = " > ".join(context_parts)
        context_prefix = f"Документ: {source}\n"
        if header_context:
            context_prefix += f"Контекст: {header_context}\n"
        context_prefix += "Содержание: "

        # Обогащаем контент: "впекаем" путь в начало текста
        enriched_content = context_prefix + content
        return enriched_content, header_context

    # ---------- кэш ----------

    def _cache_path_for(self, file_hash: str) -> Path:
        # учитываем версию пайплайна в имени кэша
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

    # ---------- утилиты ----------

    def _safe_sizeof(self, f: FileLike) -> Optional[int]:
        """Пытается получить размер файла. Возвращает None для стримов неизвестной длины."""
        try:
            if isinstance(f, (str, os.PathLike)):
                p = Path(f)
                return p.stat().st_size
            # file-like
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
        """Нормализует вход: путь → BytesIO, file-like → BytesIO."""
        if isinstance(f, (str, os.PathLike)):
            p = Path(f)
            with open(p, "rb") as fh:
                data = fh.read()
            return io.BytesIO(data), p.name
        # file-like
        if hasattr(f, "read"):
            # может быть текстовый — нормализуем к bytes
            raw = f.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            return io.BytesIO(raw), getattr(f, "name", "uploaded_file")
        raise TypeError(f"Unsupported file type: {type(f)}")

    def _hash_bytes_stream(
        self, stream: io.BytesIO, block_size: int = 1024 * 1024
    ) -> str:
        """SHA256 потоково, без загрузки всего в память второй раз."""
        stream.seek(0)
        h = hashlib.sha256()
        while True:
            chunk = stream.read(block_size)
            if not chunk:
                break
            h.update(chunk)
        stream.seek(0)
        return h.hexdigest()

    def _suffix_from_name(self, name: str) -> str:
        suf = Path(name).suffix.lower()
        return suf if suf else ".bin"

    @staticmethod
    def _normalize_text(text: str) -> str:
        # простая нормализация для дедупликации
        return " ".join(text.strip().lower().split())
