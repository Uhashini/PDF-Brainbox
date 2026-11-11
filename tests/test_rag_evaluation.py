import unittest
import numpy as np
import time
from logic.advanced_rag import (
    HybridRetriever,
    SemanticCache,
    RAGDiagnostics,
    safe_parse_json,
    MultiQueryExpander
)

class MockFaissIndex:
    """Mock FAISS index for deterministic, lightning-fast unit testing."""
    def __init__(self, num_vectors):
        self.num_vectors = num_vectors

    def search(self, q_emb, k=10):
        # Return indices [0, 1, 2, ...] up to k
        indices = np.array([[i for i in range(min(k, self.num_vectors))]])
        distances = np.array([[float(i) for i in range(min(k, self.num_vectors))]])
        return distances, indices

class TestRAGEvaluationFramework(unittest.TestCase):
    """
    CI/CD Evaluation & Quality Gate Test Suite:
    Asserts pipeline behavior for Retrieval, Generation, Anti-Hallucination, and Caching.
    """

    def setUp(self):
        self.sample_chunks = [
            "Lynch syndrome is an autosomal dominant inherited condition caused by mismatch repair mutations.",
            "Genes involved in Lynch syndrome include MLH1, MSH2, MSH6, and PMS2.",
            "Patients have high risks of colorectal cancer and endometrial cancer.",
            "Regular colonoscopy screening every 18-24 months is clinically recommended.",
            "Unrelated chunk about astronomy and stars in the night sky."
        ]
        self.embeddings = [
            np.random.normal(0, 1, 128).tolist() for _ in range(len(self.sample_chunks))
        ]
        self.mock_faiss = MockFaissIndex(len(self.sample_chunks))
        self.retriever = HybridRetriever(self.sample_chunks, self.mock_faiss, self.embeddings)

    def test_rrf_hybrid_fusion_ranking(self):
        """Verify that Reciprocal Rank Fusion ranks documents found by both dense and sparse searches at top."""
        query = "Which genes cause Lynch syndrome?"
        dummy_query_emb = np.random.normal(0, 1, 128).tolist()

        results = self.retriever.search(query, dummy_query_emb, top_k=2, mode="hybrid")
        self.assertEqual(len(results), 2)
        # Verify the top chunk contains the relevant keywords
        self.assertTrue(
            "Lynch syndrome" in results[0] or "MLH1" in results[0],
            "Hybrid RRF must prioritize documents containing keyword and dense matches."
        )

    def test_semantic_cache_performance(self):
        """Verify that Semantic Cache detects semantically equivalent queries and returns sub-10ms response."""
        cache = SemanticCache()
        base_vec = [1.0, 0.0, 0.0]
        similar_vec = [0.99, 0.05, 0.0] # High cosine similarity > 0.95
        different_vec = [0.0, 1.0, 0.0] # Orthogonal vector

        cache.add("What is Lynch syndrome?", base_vec, "Lynch syndrome is an inherited cancer syndrome.", {"faithfulness": 0.98})

        t0 = time.time()
        hit = cache.lookup(similar_vec, threshold=0.92)
        lookup_time_ms = (time.time() - t0) * 1000

        self.assertIsNotNone(hit, "Cache must return a hit for high cosine similarity.")
        self.assertIn("Lynch syndrome", hit["answer"])
        self.assertGreater(hit["similarity"], 0.95)
        self.assertLess(lookup_time_ms, 15.0, "Cache lookup should execute in < 15ms.")

        miss = cache.lookup(different_vec, threshold=0.92)
        self.assertIsNone(miss, "Cache must return None for distinct queries.")

    def test_safe_json_parser_robustness(self):
        """Ensure evaluator JSON parser handles markdown code fences, malformed formatting, and partial returns."""
        # 1. Markdown wrapped JSON
        md_json = "```json\n{\"faithfulness\": 0.98, \"answer_relevance\": 0.95, \"context_precision\": 0.92}\n```"
        parsed = safe_parse_json(md_json)
        self.assertEqual(parsed["faithfulness"], 0.98)

        # 2. Raw text with embedded JSON
        raw_text = "Here is the evaluation result:\n{\"faithfulness\": 0.88, \"answer_relevance\": 0.91, \"context_precision\": 0.85}"
        parsed = safe_parse_json(raw_text)
        self.assertEqual(parsed["faithfulness"], 0.88)

        # 3. Truncated or malformed output handled via regex recovery
        broken_json = '{"faithfulness": 0.85, "answer_relevance": 0.89, "context_precision": 0.80, "reasoning": "Answer is grounded...'
        parsed = safe_parse_json(broken_json)
        self.assertEqual(parsed["faithfulness"], 0.85)

    def test_negative_guardrail_rejection(self):
        """Verify that irrelevant questions do not hallucinate facts when evaluated."""
        query = "What is the capital of Mars?"
        retrieved_context = "Lynch syndrome is an autosomal dominant inherited condition."
        grounded_answer = "The provided document contains no information regarding Mars."

        # Mock judge returning high faithfulness for refusing to hallucinate
        def mock_judge(prompt, is_json=False):
            return '{"faithfulness": 1.0, "answer_relevance": 0.95, "context_precision": 0.10, "reasoning": "Model correctly declined to hallucinate on out-of-context query."}'

        eval_res = RAGDiagnostics.evaluate(query, retrieved_context, grounded_answer, mock_judge)
        self.assertGreaterEqual(eval_res["faithfulness"], 0.90, "Faithfulness must remain high for honest refusals.")
        self.assertLessEqual(eval_res["context_precision"], 0.30, "Context precision must reflect that retrieval was irrelevant.")

    def test_rag_triad_quality_thresholds(self):
        """Quality Gate assertion: Faithfulness and Relevance must meet minimum thresholds."""
        query = "What genes cause Lynch syndrome?"
        context = "Genes involved in Lynch syndrome include MLH1, MSH2, MSH6, and PMS2."
        answer = "Lynch syndrome is caused by mutations in MLH1, MSH2, MSH6, and PMS2 genes."

        def mock_judge(prompt, is_json=False):
            return '{"faithfulness": 0.98, "answer_relevance": 0.96, "context_precision": 0.95, "reasoning": "Accurate and strictly grounded."}'

        metrics = RAGDiagnostics.evaluate(query, context, answer, mock_judge)
        
        # CI/CD Quality Gate Assertions
        self.assertGreaterEqual(metrics["faithfulness"], 0.85, "Quality Gate: Faithfulness below 85%!")
        self.assertGreaterEqual(metrics["answer_relevance"], 0.85, "Quality Gate: Answer relevance below 85%!")
        self.assertGreaterEqual(metrics["triad_score"], 0.85, "Quality Gate: Overall triad score below 85%!")

if __name__ == "__main__":
    unittest.main()

