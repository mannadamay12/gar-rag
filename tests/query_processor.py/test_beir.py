from src.evaluation.beir_evaluator import BEIREvaluator
import json
def test_beir_evaluation():
    # Test with small number of queries first
    evaluator = BEIREvaluator(dataset_name="nfcorpus")
    stats = evaluator.evaluate_queries(num_queries=10)
    
    assert stats['total_queries'] == 10
    assert 'difficulty_distribution' in stats
    assert 'intent_distribution' in stats

if __name__ == "__main__":
    # For full evaluation
    evaluator = BEIREvaluator()
    stats = evaluator.evaluate_queries()
    print("Evaluation Statistics:")
    print(json.dumps(stats, indent=2))