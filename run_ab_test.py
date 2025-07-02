import datetime
import os

from dotenv import load_dotenv
from langchain.smith import RunEvalConfig
from langsmith import Client

# --- Импортируем цепочки кандидаты ---
# from src.chain import create_final_rag_chain
# from src.hybrid_chain import create_hybrid_rag_chain
# from src.sentence_window_chain import create_sentence_window_chain
# from src.hyde_chain import create_hyde_rag_chain
# from src.ultimate_chain import create_ultimate_rag_chain
from src.final_chain import create_final_hybrid_chain

# Убираем предупреждения о параллелизме
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# sentence_window_chain = create_sentence_window_chain()
# hyde_chain = create_hyde_rag_chain()

load_dotenv()


# Указываем наш основной датасет для тестов (Название датасета в LangSmith)
# DATASET_NAME = "safety-docs-qa"  #
DATASET_NAME = "golden-questions"


def main():
    """
    Основная функция для запуска A/B теста.
    """
    client = Client()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Просим LangSmith автоматически оценить каждый ответ по трем критериям.
    evaluation_config = RunEvalConfig(
        evaluators=[
            # Судья №1: Проверяет ответ на корректность по сравнению с эталоном (Correctness)
            # Задает вопрос: "Совпадает ли ответ по смыслу с эталонным ответом из датасета?"
            "qa",
            # Судья №2: Проверяет ответ на галлюцинации (Faithfulness / Groundedness)
            # Задает вопрос: "Основан ли ответ строго на предоставленном контексте?"
            "context_qa",
            # Судья №3: Оценивает смысловое соответствие эталону по числовой шкале (Correctness Score)
            # Сравнивает ответ модели с эталонным (labeled) ответом и выставляет оценку (score) от 0 до 1.
            "labeled_score_string",
        ]
    )

    # final_chain_for_test, _ = create_final_rag_chain()

    # Словарь с "кандидатами" на тестирование
    candidate_chains = {
        # "langchain_native_rag": final_chain_for_test, # Ваша оригинальная цепочка
        # # "langchain_llama_index_hybrid": create_hybrid_rag_chain(), # Наша гибридная цепочка
        # # "sentence_window_llamaindex": sentence_window_chain,
        # "hyde_advanced_rag": hyde_chain,
        # "ultimate_rag": create_ultimate_rag_chain(),
        "final_rag": create_final_hybrid_chain()  # Новая ультимативная цепочка
    }

    print(
        f"Запускаем A/B тест для {len(candidate_chains)} моделей на датасете '{DATASET_NAME}'..."
    )

    for chain_name, chain in candidate_chains.items():
        print(f"--- Тестируем модель: {chain_name} ---")
        try:
            project_name = f"test-{chain_name}-{timestamp}"
            # Запускаем тест для каждой цепочки
            client.run_on_dataset(
                dataset_name=DATASET_NAME,
                llm_or_chain_factory=lambda: chain,  # Фабрика для создания объекта цепочки
                project_name=project_name,  # Создаем отдельный проект для каждого теста
                evaluation=evaluation_config,  # <-- ПЕРЕДАЕМ "СУДЕЙ"
                concurrency_level=1,  # Выполняем по одному, чтобы избежать проблем с API
            )
            print(
                f"Тестирование для '{chain_name}' успешно запущено. Результаты скоро появятся в LangSmith."
            )
        except Exception as e:
            print(f"Ошибка при тестировании '{chain_name}': {e}")


if __name__ == "__main__":
    main()
