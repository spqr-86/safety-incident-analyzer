# Codebase Analysis

## ⚡ TL;DR

**Что это:** RAG-система с мульти-агентным контролем качества (LangGraph) и гибридным поиском (Chroma + BM25 + FlashRank).

**Топ-3 команды:**
1. `python index.py` — индексация документов
2. `streamlit run app.py` — запуск интерфейса
3. `pytest` — запуск тестов

**Ключевые файлы:** `agents/workflow.py`, `src/final_chain.py`, `index.py`

---

## ⚙️ Как это работает

Система использует **LangGraph** для оркестрации агентов. Процесс обработки запроса выглядит так:

```mermaid
sequenceDiagram
    participant User
    participant Workflow as AgentWorkflow
    participant Relevance as RelevanceChecker
    participant Research as ResearchAgent
    participant Verify as VerificationAgent

    User->>Workflow: Запрос
    Workflow->>Relevance: Проверка релевантности
    alt Irrelevant
        Relevance-->>Workflow: CANNOT_ANSWER
        Workflow-->>User: Отказ
    else Relevant
        Relevance-->>Workflow: CAN_ANSWER / PARTIAL
        loop Research Loop (Max 3)
            Workflow->>Research: Поиск и генерация (CoT + Normative Filter)
            Note right of Research: Анализ Found/Not Found, ответ в <answer>
            Research-->>Workflow: Черновик ответа + Ход мыслей
            Workflow->>Verify: Проверка фактов и логики
            alt Good
                Verify-->>Workflow: CORRECT
                Workflow-->>User: Финальный ответ + Chain-of-Thought
            else Bad
                Verify-->>Workflow: REVISE (Замечания + Logic Leap Check)
                Note right of Workflow: Возврат на доработку (increment loops)
            end
        end
        alt Max Loops Reached
            Workflow-->>User: Ответ (Partial) + Оговорка о верификации
        end
    end
```

### Основные этапы

| Step | Component | Action |
|------|-----------|--------|
| 1. Ingestion | `index.py` | Загрузка и индексация документов в ChromaDB. См. [DATA_PIPELINE.md](../DATA_PIPELINE.md) |
| 2. Retrieval | `src/final_chain.py` | Гибридный поиск с `ApplicabilityAwareRetriever` (расширение запроса для общих норм) |
| 3. Relevance Check | `agents/relevance_checker.py` | Классификация вопроса с использованием CoT (CAN_ANSWER/PARTIAL/NO) |
| 4. Generation | `agents/research_agent.py` | Фильтрация атрибутов (Normative Accuracy), генерация ответа в `<answer>` |
| 5. Verification | `agents/verification_agent.py` | Проверка на галлюцинации и "Logic Leaps", возврат фидбека (макс 3 итерации) |

---

## 🛡️ Философия "Нормативной точности"

Система действует как строгое "нормативное зеркало". Она не пытается угадать намерения пользователя, а отражает только те факты, которые явно прописаны в нормах:

1.  **Фильтрация атрибутов**: Разделение запроса на нормативно значимые термины ( Found) и "бытовой шум" (Not Found).
2.  **Запрет на домысливание**: Мы не приравниваем "бухгалтера" к "офису", если это не написано в документе. Мы отвечаем: "Для бухгалтера норм нет. Общие нормы для ПЭВМ такие: ...".
3.  **Стабилизация графа**: Счетчик `loops` вынесен в узел графа, что гарантирует сохранение состояния и корректную остановку цикла после 3 попыток.

---

## 🚀 Первые шаги разработчика

### Запуск локально
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python index.py          # Индексация базы знаний
streamlit run app.py     # Запуск UI
```

### Типовые задачи

**Хочу понять [RAG]:**
Изучите `src/final_chain.py`, чтобы разобраться в механике поиска и генерации.

**Хочу понять [Agents]:**
Смотрите `agents/workflow.py` для понимания графа переходов состояний.

**Добавить новую фичу:**
1. Создайте нового агента в `agents/`.
2. Добавьте узел (node) в граф в `agents/workflow.py`.

**Исправить ошибку:**
1. Проверьте логи ошибок в `analysis/error_reports`.
2. Напишите воспроизводящий тест в `tests/`.

**Написать тест:**
Используйте `pytest`. Тесты находятся в папке `tests/`. См. подробнее в [testing.md](../guides/testing.md).

**Хочу проверить документацию:**
См. [testing.md](../guides/testing.md) для инструкции по проверке ссылок и актуальности.

### Развертывание (Deploy)
- **Streamlit Cloud:** Подключите репозиторий GitHub и укажите `app.py`.
- **Docker:** Используйте `Dockerfile` для контейнеризации.

---

## 🗺 Карта кодовой базы

| Директория | Описание |
|------------|----------|
| `agents/` | Логика мульти-агентной системы (LangGraph). Содержит `RelevanceChecker`, `ResearchAgent`, `VerificationAgent`. |
| `prompts/` | Централизованное хранилище промптов (Jinja2) и реестр версий (`registry.yaml`). |
| `src/` | Ядро RAG-логики: цепочки (`final_chain.py`), работа с векторной БД (`vector_store.py`), фабрики LLM. |
| `config/` | Настройки приложения и переменных окружения (`settings.py`). |
| `eval/` | Скрипты для оценки качества ответов (DeepEval, Ragas). |
| `tests/` | Юнит и интеграционные тесты. |
| `analysis/` | Отчеты об ошибках и логи работы. |

---

## 🔬 Углублённо

<details>
<summary><b>Алгоритм работы (20 шагов)</b></summary>

1. Пользователь вводит запрос.
2. Система инициализирует состояние графа.
3. `RelevanceChecker` анализирует запрос.
4. LLM определяет класс запроса (Relevant/Not).
5. Если не релевантно -> ответ-заглушка.
6. Если релевантно -> передача управления `ResearchAgent`.
7. `ResearchAgent` формирует поисковый запрос.
8. Retriever выполняет поиск в ChromaDB (векторный).
9. Retriever выполняет поиск BM25 (ключевые слова).
10. Результаты объединяются (Ensemble Retriever).
11. FlashRank переранжирует результаты для повышения точности.
12. Топ-K документов передаются в контекст.
13. LLM генерирует черновик ответа по контексту.
14. Ответ передается `VerificationAgent`.
15. `VerificationAgent` сверяет утверждения с контекстом.
16. Оценка качества (Correctness/Faithfulness).
17. Если оценка высокая -> вывод ответа.
18. Если есть ошибки -> формирование фидбека.
19. Возврат к `ResearchAgent` (цикл исправления, макс. 2 раза).
20. Финальный вывод результата пользователю.

</details>

### Known Issues / TODO
- [ ] Оптимизация скорости FlashRank (задержка на CPU).
- [ ] Улучшение обработки таблиц в PDF документах.
- [ ] Добавление памяти диалога (Chat History) в LangGraph.
- [ ] Расширение набора тестов для агентов.

### Prompt Management System

Система управления промптами позволяет редактировать поведение LLM без изменения кода.
- **Хранение:** Все промпты лежат в `prompts/` (Jinja2 шаблоны).
- **Реестр:** `prompts/registry.yaml` управляет версиями.
- **Pinning:** Версию любого промпта можно переопределить через переменную окружения `PROMPT_{ID}_VERSION`.

Подробнее см. в руководстве: [Prompt Management Guide](../guides/prompt-management.md).

### Relevance Checker
Первый рубеж обороны. Классифицирует вопрос пользователя на три категории:
- `CAN_ANSWER`: Вопрос по теме охраны труда.
- `PARTIAL`: Частично по теме, требует уточнения.
- `NO`: Вопрос не по теме (спам, chit-chat).

### Verification Agent
Критик, который проверяет сгенерированный ответ на соответствие найденным документам (контексту).
- Если галлюцинаций нет -> ответ отдается пользователю.
- Если есть ошибки -> отправляет на перегенерацию с указанием, что исправить.
