from typing import Dict, List, Type
from ..query_processor.preprocessor import QueryProcessor
from ..models.base_model import BaseLanguageModel
from ..models.llama import LlamaModel
from ..models.falcon import FalconModel
from ..models.flan_t5 import FlanT5Model
from .prompt_templates import PromptTemplates
from ..utils.logger import setup_logger
from .strategies import (
    EnhancementStrategy,
    SimpleExpansionStrategy,
    DomainSpecificStrategy,
    ComplexQueryStrategy
)

logger = setup_logger("gar_enhancer")

class GAREnhancer:
    def __init__(self, model_type: str = "flan-t5"):
        self.query_processor = QueryProcessor()
        self.model = self._get_model(model_type)
        self.strategies = {
            'easy': SimpleExpansionStrategy(),
            'moderate': DomainSpecificStrategy(),
            'difficult': ComplexQueryStrategy(),
            'very_difficult': ComplexQueryStrategy()
        }

    def _get_model(self, model_type: str) -> BaseLanguageModel:
        """Get appropriate language model based on type"""
        models = {
            'flan-t5': FlanT5Model,
            'llama': LlamaModel,
            'falcon': FalconModel
        }
        return models[model_type]()

    def enhance_query(self, query: str) -> Dict:
        """
        Enhance query using GAR approach
        """
        try:
            # Process original query
            query_info = self.query_processor.process_query(query)

            difficulty = query_info['difficulty']['difficulty_level']
            strategy = self.strategies[difficulty]
            
            enhanced = strategy.enhance(query_info, self.model)
            
            return {
                'original_query': query,
                'enhanced_query': enhanced['enhanced_query'],
                'strategy_used': enhanced['strategy'],
                'query_info': query_info
            }
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            raise