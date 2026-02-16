# tests/v7/test_nlp_core.py
import unittest
import numpy as np

from src.v7.state_types import Doc, ScoredDoc

# Import the functions to be tested
from src.v7.nlp_core import (
    extract_keywords,
    BM25Index,
    rrf_merge,
    mmr_select,
)

class TestNlpCore(unittest.TestCase):

    def test_extract_keywords(self):
        text = "Проверка извлечения ключевых слов из этого текста, без предлогов и союзов."
        expected = ["проверка", "извлечение", "ключевой", "слово", "это", "текст", "предлог", "союз"]
        actual = extract_keywords(text)
        self.assertCountEqual(expected, actual)

    def test_bm25_index(self):
        docs = [
            Doc(id="doc1", text="Первый документ о кошках.", metadata={}),
            Doc(id="doc2", text="Второй документ про собак.", metadata={}),
            Doc(id="doc3", text="Третий документ про кошек и собак.", metadata={}),
        ]
        
        bm25 = BM25Index()
        bm25.build(docs)
        
        query_tokens = extract_keywords("кошки")
        results = bm25.query(query_tokens, top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "doc3")
        self.assertEqual(results[1].id, "doc1")

    def test_rrf_merge(self):
        ranking1 = [
            ScoredDoc(id="docA", text="", metadata={}, score=0.9),
            ScoredDoc(id="docB", text="", metadata={}, score=0.8),
            ScoredDoc(id="docC", text="", metadata={}, score=0.7),
        ]
        ranking2 = [
            ScoredDoc(id="docB", text="", metadata={}, score=0.95),
            ScoredDoc(id="docA", text="", metadata={}, score=0.85),
            ScoredDoc(id="docD", text="", metadata={}, score=0.75),
        ]
        
        merged = rrf_merge([ranking1, ranking2], k=1)
        
        self.assertEqual(len(merged), 4)
        # docA and docB should be the top 2
        top_2_ids = {merged[0].id, merged[1].id}
        self.assertEqual(top_2_ids, {"docA", "docB"})
        self.assertTrue(merged[0].score > merged[2].score)
        self.assertTrue(merged[1].score > merged[2].score)


    def test_mmr_select_placeholder(self):
        docs = [Doc(id=f"doc{i}", text=f"text {i}", metadata={}) for i in range(10)]
        query_emb = np.random.rand(1, 10) # Dummy embedding
        
        selected = mmr_select(docs, query_emb, lambda_=0.5, k=5)
        
        self.assertEqual(len(selected), 5)
        self.assertEqual([d.id for d in selected], ["doc0", "doc1", "doc2", "doc3", "doc4"])

if __name__ == "__main__":
    unittest.main()
