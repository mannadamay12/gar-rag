# src/evaluation/model_comparator.py
from typing import List, Dict
import pandas as pd
from ..gar.enhancer import GAREnhancer
from ..utils.logger import setup_logger

logger = setup_logger("model_comparator")

class ModelComparator:
    def __init__(self, models: List[str] = ['flan-t5', 'llama', 'falcon']):
        self.enhancers = {
            model: GAREnhancer(model_type=model)
            for model in models
        }

    def compare_models(self, queries: List[str]) -> pd.DataFrame:
        """Compare enhancement results across models"""
        results = []
        
        for query in queries:
            for model_name, enhancer in self.enhancers.items():
                try:
                    enhanced = enhancer.enhance_query(query)
                    
                    result = {
                        'original_query': query,
                        'model': model_name,
                        'enhanced_query': enhanced['enhanced_query'],
                        'strategy_used': enhanced['strategy_used'],
                        'difficulty': enhanced['query_info']['difficulty']['difficulty_level']
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error with model {model_name} on query '{query}': {str(e)}")
        
        return pd.DataFrame(results)

# Usage example:
if __name__ == "__main__":
    test_queries = [
        "vitamin D benefits",
        "complex quantum mechanics principles",
        "nearest coffee shop"
    ]
    
    comparator = ModelComparator()
    results = comparator.compare_models(test_queries)
    print(results.to_string())