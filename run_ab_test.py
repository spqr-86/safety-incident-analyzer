import datetime

from langsmith import Client
from dotenv import load_dotenv

# --- Импортируем наши цепочки ---
# Важно: Нам нужно немного изменить наши файлы с цепочками,
# чтобы они не выполняли код при импорте, а предоставляли функцию для создания цепочки.
# Мы сделаем это в следующем шаге.
# from src.chain import create_final_rag_chain
# from src.hybrid_chain import create_hybrid_rag_chain # Предполагаем, что такая функция будет
# from src.sentence_window_chain import create_sentence_window_chain
# from src.hyde_chain import create_hyde_rag_chain
# from src.ultimate_chain import create_ultimate_rag_chain
from src.final_chain import create_final_hybrid_chain

# sentence_window_chain = create_sentence_window_chain()
# hyde_chain = create_hyde_rag_chain()

load_dotenv()

# --- Настройка ---
# DATASET_NAME = "safety-docs-qa"  # Название датасета, которое вы создали в LangSmith
DATASET_NAME = "golden-questions"

def main():
    """
    Основная функция для запуска A/B теста.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    client = Client()

    # final_chain_for_test, _ = create_final_rag_chain()

    # Словарь с нашими "кандидатами" на тестирование
    candidate_chains = {
        # "langchain_native_rag": final_chain_for_test, # Ваша оригинальная цепочка
        # # "langchain_llama_index_hybrid": create_hybrid_rag_chain(), # Наша гибридная цепочка
        # # "sentence_window_llamaindex": sentence_window_chain,
        # "hyde_advanced_rag": hyde_chain,
        # "ultimate_rag": create_ultimate_rag_chain(),
        "final_rag": create_final_hybrid_chain() # Новая ультимативная цепочка
    }

    print(f"Запускаем A/B тест для {len(candidate_chains)} моделей на датасете '{DATASET_NAME}'...")

    for chain_name, chain in candidate_chains.items():
        print(f"--- Тестируем модель: {chain_name} ---")
        try:
            project_name = f"test-{chain_name}-{timestamp}"
            # Запускаем тест для каждой цепочки
            client.run_on_dataset(
                dataset_name=DATASET_NAME,
                llm_or_chain_factory=lambda: chain, # Фабрика для создания объекта цепочки
                project_name=project_name, # Создаем отдельный проект для каждого теста
                concurrency_level=1, # Выполняем по одному, чтобы избежать проблем с API
            )
            print(f"Тестирование для '{chain_name}' успешно запущено. Результаты скоро появятся в LangSmith.")
        except Exception as e:
            print(f"Ошибка при тестировании '{chain_name}': {e}")

if __name__ == "__main__":
    main()
