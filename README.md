# 🛡️ AI Safety Compliance Assistant

> Интеллектуальная система анализа нормативной документации по охране труда с использованием RAG-технологий

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0+-orange.svg)](https://chromadb.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-red.svg)](https://streamlit.io)

## 🎯 О проекте

AI Safety Compliance Assistant — это умная система для работы с нормативными документами по охране труда. Система использует передовые технологии искусственного интеллекта для быстрого поиска и анализа требований безопасности в больших объемах документации.

Проект сочетает RAG (Retrieval-Augmented Generation) с многоагентным MAS-подходом, чтобы давать точные, проверенные ответы на основе **ГОСТ, СНиП и внутренних документов компании**.

## 🚀 Демо

**Попробовать работающее приложение можно здесь:**

**[➡️ https://safety-incident-analyzer-sefffd3s4bnafeezqfpmv7.streamlit.app/] ⬅️**

![Скриншот приложения](assets/screenshot.png)

**Ключевые возможности:**
- 🔍 Семантический поиск по документам
- 💬 Интерактивный чат с контекстом беседы
- 📄 Автоматическая обработка PDF документов
- ⚡ Быстрые и точные ответы на вопросы по ОТ
- 🎯 Ранжирование результатов по релевантности

## 🚀 Быстрый старт

### Предварительные требования
- Python 3.11+
- OpenAI API ключ

### 1. Установка

```bash
# Клонируем репозиторий
git clone https://github.com/your-username/safety-incident-analyzer.git
cd safety-incident-analyzer

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 2. Настройка окружения

Создайте файл `.env` в корне проекта:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Подготовка данных

```bash
# Поместите ваши PDF документы в папку data/
mkdir -p data
# Скопируйте PDF файлы в папку data/

# Создайте векторную базу данных
python index.py
```

### 4. Запуск приложения

```bash
streamlit run app.py
```

Приложение будет доступно по адресу: `http://localhost:8501`

## 🧭 Архитектура (RAG + MAS)

```mermaid
flowchart LR
    subgraph Ingestion[Индексация / Предобработка]
        A[Документы: СНиП, ГОСТ, СП, внутренние регламенты] --> B[Docling → Markdown]
        B --> C[Чанкинг (chunk_size, overlap)]
        C --> D[Эмбеддинги (API/Local)]
        C --> E[BM25 индекс]
        D --> F[ChromaDB (persist)]
        E --> H[(BM25)]
    end

    subgraph App[Приложение (Streamlit)]
        Q[Вопрос пользователя] --> R[Гибридный ретривер]
        R -->|weights,k| R1[Векторный поиск (Chroma)]
        R -->|weights,k| R2[BM25]
        R1 --> RR[FlashRank Re-Ranker]
        R2 --> RR
        RR --> P[Топ-фрагменты (контекст)]
    end

    subgraph MAS[MAS Workflow (LangGraph)]
        P --> RC[RelevanceChecker]
        RC -->|релевантно| RS[ResearchAgent (LLM)]
        RC -->|не релевантно| X[[Корректный отказ]]
        RS --> V[VerificationAgent (LLM-as-Judge)]
        V -->|NO| L{{Decision Layer}}
        L -->|повтор| RS
        V -->|OK| OUT[Финальный ответ + ссылки]
    end

    OUT --> U[(UI: Ответ + Источники + Отчёт верификации)]
    X --> U


### Последовательность обработки запроса:

sequenceDiagram
    participant User as Пользователь
    participant UI as Streamlit UI
    participant Ret as Гибридный ретривер
    participant Rerank as FlashRank
    participant MAS as LangGraph (MAS)
    participant R as ResearchAgent
    participant V as VerificationAgent
    participant DB as Chroma + BM25

    User->>UI: Вопрос
    UI->>Ret: invoke(question)
    Ret->>DB: семантический + BM25
    DB-->>Ret: кандидаты (фрагменты)
    Ret->>Rerank: rerank(candidates)
    Rerank-->>UI: top-k фрагментов

    UI->>MAS: start(state: question, docs)
    MAS->>MAS: RelevanceChecker
    alt Релевантно
        MAS->>R: generate(docs)
        R-->>MAS: draft_answer
        MAS->>V: check(draft_answer, docs)
        V-->>MAS: verification_report
        alt Не подтверждено
            MAS->>R: refine & regenerate
            R-->>MAS: new_draft
            MAS->>V: re-check
        end
        MAS-->>UI: финальный ответ + отчёт
    else Не релевантно
        MAS-->>UI: корректный отказ
    end

    UI-->>User: Ответ + Источники + Верификация


## 📁 Структура проекта

```
safety-incident-analyzer/
├── app.py                      # Streamlit UI (RAG-режим + MAS-режим)
├── index.py                    # Индексация документов в Chroma
├── requirements.txt
├── .env.example
├── config/
│   ├── constants.py
│   └── settings.py
├── src/
│   ├── file_handler.py         # Docling → Markdown → Split → Cache/Dedupe
│   ├── vector_store.py         # Chroma + embeddings + совместимость
│   ├── llm_factory.py          # LLM/Embeddings провайдеры (OpenAI/GigaChat/Local)
│   ├── final_chain.py          # Гибридный retriever + FlashRank + prompt
│   ├── agents/
│   │   ├── workflow.py         # LangGraph: Relevance→Research→Verification
│   │   ├── relevance_checker.py
│   │   ├── research_agent.py
│   │   └── verification_agent.py
│   └── retriever/
│       └── builder.py          # BM25 + Chroma + Ensemble
└── README.md
```

## ⚙️ Конфигурация

Основные настройки в `config.py`:

```python
# Модели OpenAI
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TEMPERATURE = 0.0

# Параметры чанкинга
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Пути к данным
CHROMA_DB_PATH = "chroma_db"
SOURCE_DOCS_PATH = "data"
```

## 🔧 Основные функции

### Индексация документов
```bash
python index.py
```
- Сканирует папку `data/` на наличие PDF файлов
- Разбивает документы на чанки
- Создает векторные представления
- Сохраняет в ChromaDB

### Запуск веб-интерфейса
```bash
streamlit run app.py
```
- Интерактивный чат-интерфейс
- Контекстные диалоги
- Отображение источников информации
- Стриминг ответов в реальном времени

## 🛠️ Технологический стек

- **🐍 Python 3.11+** - основной язык разработки
- **🦜 LangChain 0.3+** - фреймворк для RAG
- **🎨 ChromaDB 1.0+** - векторная база данных
- **🤖 OpenAI API** - языковые модели и эмбеддинги
- **⚡ Streamlit 1.46+** - веб-интерфейс
- **📄 PyPDF** - парсинг PDF документов
- **🎯 FlashRank** - ре-ранжирование результатов

## 📊 Возможности системы

### Обработка документов
- ✅ Поддержка PDF форматов
- ✅ Автоматическое разбиение на чанки
- ✅ Сохранение метаданных (источник, страница)
- ✅ Инкрементальное обновление базы

### Поиск и извлечение
- ✅ Семантический поиск по векторам
- ✅ Контекстная компрессия результатов
- ✅ Ре-ранжирование с помощью FlashRank
- ✅ Поддержка истории диалога

### Пользовательский интерфейс
- ✅ Интуитивный чат-интерфейс
- ✅ Стриминг ответов
- ✅ Отображение источников
- ✅ Адаптивный дизайн

## 🧪 Примеры использования

### Поиск требований безопасности
```
Q: Каковы виды обучения по охране труда?
A: Согласно документации, обучение по охране труда осуществляется в ходе проведения:
   а) инструктажей по охране труда
   б) стажировки на рабочем месте
   в) обучения по оказанию первой помощи пострадавшим
   г) обучения по использованию средств индивидуальной защиты
   д) обучения по охране труда у работодателя...
```

### Анализ нормативных требований
```
Q: Какова периодичность обучения по охране труда?
A: В зависимости от категории работников обучение проводится:
   - Руководители и специалисты: не реже одного раза в 3 года
   - Работники рабочих профессий: согласно программам обучения...
```

## 🚧 Развитие проекта

### Текущая версия (v0.1)
- [x] Базовая RAG-система
- [x] Парсинг PDF документов
- [x] Streamlit интерфейс
- [x] Контекстные диалоги
- [x] Ре-ранжирование результатов

### Планы развития (v0.2)
- [ ] Поддержка дополнительных форматов (DOCX, HTML)
- [ ] Расширенная аналитика использования
- [ ] API эндпоинты для интеграции
- [ ] Улучшенные метрики качества

### Долгосрочные планы (v1.0)
- [ ] Мультимодальная обработка (изображения, таблицы)
- [ ] Интеграция с корпоративными системами
- [ ] Автоматическое обновление документов
- [ ] Продвинутая система аналитики


## 👨‍💻 Автор

**Петр Балдаев** - AI/ML Engineer
- GitHub: [@spqr-86](https://github.com/spqr-86)
- LinkedIn: [petr-baldaev](https://linkedin.com/in/petr-baldaev-b1252b263/)
- Email: petr.baldaev.ds@gmail.com

---

⭐ **Поставьте звезду, если проект был полезен!**
