from src.gar.enhancer import GAREnhancer
from src.evaluation.gar_evaluator import GAREvaluator

# Single query enhancement
enhancer = GAREnhancer()
result = enhancer.enhance_query("electric car environmental impact")
print(result['enhanced_query'])

# Evaluation
# evaluator = GAREvaluator()
# results_df = evaluator.evaluate(num_queries=10)
# print(results_df.head())