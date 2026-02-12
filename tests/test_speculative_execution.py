import time
import pytest
from concurrent.futures import ThreadPoolExecutor


def slow_router():
    time.sleep(1.0)
    return "rag_simple"


def slow_search():
    time.sleep(1.0)
    return ["doc1", "doc2"]


def test_parallel_execution():
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        f1 = executor.submit(slow_router)
        f2 = executor.submit(slow_search)
        route = f1.result()
        docs = f2.result()
    end = time.perf_counter()

    duration = end - start
    print(f"Duration: {duration:.2f}s")

    assert duration < 1.5
    assert route == "rag_simple"
    assert len(docs) == 2
