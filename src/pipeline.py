# src/pipeline.py
from typing import Dict
from .gar.enhancer import GAREnhancer
from .rag.pipeline import RAGPipeline

class SearchPipeline:
    def __init__(self):
        self.gar_enhancer = GAREnhancer()
        self.rag_pipeline = RAGPipeline()
        
    def process_query(self, query: str) -> Dict:
        """Process query through GAR and RAG pipeline"""
        # Enhance query using GAR
        enhanced = self.gar_enhancer.enhance_query(query)
        
        # Process through RAG pipeline
        result = self.rag_pipeline.process(
            query=query,
            enhanced_query=enhanced['enhanced_query']
        )
        
        return {
            'answer': result['answer'],
            'sources': result['sources'],
            'metadata': {
                **result['metadata'],
                'query_enhancement': enhanced['strategy_used']
            }
        }