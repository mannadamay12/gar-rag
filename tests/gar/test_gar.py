import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from src.gar.enhancer import GAREnhancer

class GARTestingFramework:
    def __init__(self, dataset_name="nfcorpus", data_path="datasets", model_type="flan-t5"):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.gar_enhancer = GAREnhancer(model_type=model_type)

    def load_dataset(self):
        print(f"Loading local dataset: {self.dataset_name}")
        dataset_path = f"{self.data_path}/{self.dataset_name}"
        data_loader = GenericDataLoader(dataset_path)
        corpus, queries, qrels = data_loader.load()
        return corpus, queries, qrels

    def _prepare_corpus(self, corpus):
        """Convert corpus to format needed for BM25"""
        tokenized_corpus = []
        doc_ids = []
        for doc_id, doc in corpus.items():
            tokenized_corpus.append(doc['text'].split())
            doc_ids.append(doc_id)
        return tokenized_corpus, doc_ids

    def retrieve_baseline(self, corpus, queries, k=10):
        """Retrieve documents without query enhancement"""
        print("Running baseline retrieval (BM25)...")
        tokenized_corpus, doc_ids = self._prepare_corpus(corpus)
        bm25 = BM25Okapi(tokenized_corpus)
        
        results = {}
        for qid, query in queries.items():
            tokenized_query = query.split()
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Create sorted results dictionary
            query_results = {}
            for doc_id, score in zip(doc_ids, doc_scores):
                query_results[doc_id] = score
            
            # Sort by score and take top k
            sorted_results = dict(sorted(query_results.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:k])
            results[qid] = sorted_results
            
        return results

    def retrieve_with_gar(self, corpus, queries, k=10):
        """Retrieve documents with GAR-enhanced queries"""
        print("Enhancing queries with GAR...")
        enhanced_queries = {}
        for qid, query in queries.items():
            try:
                enhanced = self.gar_enhancer.enhance_query(query)
                enhanced_queries[qid] = enhanced['enhanced_query']
            except Exception as e:
                print(f"Error enhancing query {qid}: {e}")
                enhanced_queries[qid] = query  # Fallback to original query if enhancement fails

        print("Running retrieval with enhanced queries (BM25)...")
        tokenized_corpus, doc_ids = self._prepare_corpus(corpus)
        bm25 = BM25Okapi(tokenized_corpus)
        
        results = {}
        for qid, query in enhanced_queries.items():
            tokenized_query = query.split()
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Create sorted results dictionary
            query_results = {}
            for doc_id, score in zip(doc_ids, doc_scores):
                query_results[doc_id] = score
            
            # Sort by score and take top k
            sorted_results = dict(sorted(query_results.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:k])
            results[qid] = sorted_results
            
        return results

    def evaluate(self, results, qrels):
        """Evaluate retrieval results using BEIR metrics"""
        print("Evaluating retrieval results...")
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels, results, [10])
        return metrics

    def run_tests(self):
        """Run all tests and compare baseline with GAR"""
        try:
            # Load dataset
            corpus, queries, qrels = self.load_dataset()
            
            if not corpus or not queries or not qrels:
                raise ValueError("Failed to load dataset properly")

            # Baseline retrieval
            baseline_results = self.retrieve_baseline(corpus, queries)
            if not baseline_results:
                raise ValueError("Baseline retrieval failed")
            
            baseline_metrics = self.evaluate(baseline_results, qrels)

            # GAR-enhanced retrieval
            gar_results = self.retrieve_with_gar(corpus, queries)
            if not gar_results:
                raise ValueError("GAR-enhanced retrieval failed")
                
            gar_metrics = self.evaluate(gar_results, qrels)

            # Display comparison
            comparison = {
                "Metric": list(baseline_metrics.keys()),
                "Baseline": list(baseline_metrics.values()),
                "GAR-Enhanced": list(gar_metrics.values())
            }
            comparison_df = pd.DataFrame(comparison)
            print("\nComparison of Metrics:")
            print(comparison_df)
            return comparison_df

        except Exception as e:
            print(f"Error during testing: {e}")
            print(f"Full error details: ", e.__class__.__name__)
            return None

# Example usage
if __name__ == "__main__":
    framework = GARTestingFramework(dataset_name="nfcorpus", model_type="flan-t5")
    comparison_df = framework.run_tests()
    if comparison_df is not None:
        comparison_df.to_csv("gar_testing_results.csv", index=False)
    else:
        print("Testing failed - no results to save")
