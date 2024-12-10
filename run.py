from src.evaluation.beir_evaluator import BEIREvaluator

# Test with NFCorpus dataset
evaluator = BEIREvaluator(dataset_name="nfcorpus")
stats = evaluator.evaluate_queries(num_queries=100)  # Start with subset
print(stats)