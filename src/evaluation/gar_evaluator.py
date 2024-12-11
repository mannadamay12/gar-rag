from typing import Dict, List
import pandas as pd
from ..gar.enhancer import GAREnhancer
from .beir_evaluator import BEIREvaluator
from ..utils.logger import setup_logger

logger = setup_logger("gar_evaluator")

class GAREvaluator:
    def __init__(self):
        self.enhancer = GAREnhancer()
        self.beir_evaluator = BEIREvaluator()

    def evaluate(self, num_queries: int = None) -> Dict:
        """
        Evaluate GAR enhancement on BEIR queries
        """
        _, queries, _ = self.beir_evaluator.load_dataset()
        
        if num_queries:
            queries = dict(list(queries.items())[:num_queries])
        
        results = []
        
        for qid, query_text in queries.items():
            enhanced = self.enhancer.enhance_query(query_text)
            
            result = {
                'query_id': qid,
                'original_query': query_text,
                'enhanced_query': enhanced['enhanced_query'],
                'num_expansions': len(enhanced['expansions']),
                'difficulty_level': enhanced['query_info']['difficulty']['difficulty_level']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)