import unittest
from llm import ModelFactory, SentenceTransformerEmbedding, FaissIndexSingleton, FitzPDFReader

class TestPatterns(unittest.TestCase):
    def test_factory_model(self):
        model = ModelFactory.get_model("gemma-7b-it")
        self.assertIsNotNone(model)

    def test_strategy_embedding(self):
        strategy = SentenceTransformerEmbedding("all-mpnet-base-v2")
        embedding = strategy.encode(["test cümlesi"])
        self.assertIsNotNone(embedding)

    def test_singleton_faiss(self):
        index1 = FaissIndexSingleton.get_instance(384)
        index2 = FaissIndexSingleton.get_instance(384)
        self.assertIs(index1, index2)

    def test_adapter_pdf_reader(self):
        reader = FitzPDFReader()
        # Sadece sınıfın varlığını ve metodu test ediyoruz, gerçek PDF ile test için dosya gerekir
        self.assertTrue(hasattr(reader, 'read'))

if __name__ == "__main__":
    unittest.main() 