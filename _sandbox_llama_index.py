from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

load_dotenv()

print("Начинаем процесс с LlamaIndex...")

print("Шаг 1: Загрузка документов из папки 'data/'...")
documents = SimpleDirectoryReader("./data").load_data()
print(f"Загружено {len(documents)} документов.")


print("Шаг 2: Создание индекса из документов. Это может занять некоторое время...")
index = VectorStoreIndex.from_documents(documents)
print("Индекс успешно создан.")

print("Шаг 3: Создание движка запросов и выполнение запроса...")
query_engine = index.as_query_engine()
response = query_engine.query("Какие существуют виды обучения по охране труда?")

print("\n--- Результат ---")
print("Ответ модели:")
print(response)
print("\n-----------------")
