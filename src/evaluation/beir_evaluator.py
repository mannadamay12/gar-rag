from typing import Dict, List
import json
from pathlib import Path
import pandas as pd
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from ..query_processor.preprocessor import QueryProcessor
from ..utils.logger import setup_logger

logger = setup_logger("beir_evaluator")

class BEIREvaluator:
    def __init__(self, dataset_name: str = "nfcorpus"):
        self.dataset_name = dataset_name
        self.processor = QueryProcessor()
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

    def load_dataset(self) -> tuple:
        """Load BEIR dataset"""
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            data_path = util.download_and_unzip(url, "datasets")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
            logger.info(f"Successfully loaded {self.dataset_name} dataset")
            return corpus, queries, qrels
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def evaluate_queries(self, num_queries: int = None) -> Dict:
        """
        Evaluate query processor on BEIR dataset queries
        """
        try:
            # Load dataset
            _, queries, _ = self.load_dataset()
            
            if num_queries:
                queries = dict(list(queries.items())[:num_queries])
            
            results = []
            
            # Process each query
            for qid, query_text in queries.items():
                logger.info(f"Processing query {qid}: {query_text}")
                
                # Process query
                processed = self.processor.process_query(query_text)
                
                # Extract key metrics
                result = {
                    'query_id': qid,
                    'query_text': query_text,
                    'difficulty_level': processed['difficulty']['difficulty_level'],
                    'difficulty_score': processed['difficulty']['final_score'],
                    'primary_intent': processed['intent']['primary'][0].value,
                    'intent_confidence': processed['intent']['primary'][1],
                    'num_entities': len(processed['entities']),
                    'token_count': processed['token_count']
                }
                
                results.append(result)
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(results)
            
            # Generate statistics
            stats = {
                'total_queries': len(df),
                'difficulty_distribution': df['difficulty_level'].value_counts().to_dict(),
                'intent_distribution': df['primary_intent'].value_counts().to_dict(),
                'avg_difficulty_score': df['difficulty_score'].mean(),
                'avg_token_count': df['token_count'].mean(),
                'avg_entities': df['num_entities'].mean()
            }
            
            # Save results
            self._save_results(df, stats)
            
            logger.info("Evaluation completed successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

    def _save_results(self, df: pd.DataFrame, stats: Dict):
        """Save evaluation results"""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        df.to_csv(self.results_dir / f"{self.dataset_name}_results_{timestamp}.csv", index=False)
        
        # Save statistics
        with open(self.results_dir / f"{self.dataset_name}_stats_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=4)