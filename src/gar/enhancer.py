from typing import Dict, List
from ..query_processor.preprocessor import QueryProcessor
from ..models.flan_t5 import FlanT5Model
from .prompt_templates import PromptTemplates
from ..utils.logger import setup_logger
from .strategies import QueryEnhancementStrategies

logger = setup_logger("gar_enhancer")

class GAREnhancer:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.model = FlanT5Model()
        self.templates = PromptTemplates()
        self.strategies = QueryEnhancementStrategies(self.model)

    def enhance_query(self, query: str) -> Dict:
        """
        Enhance query using GAR approach
        """
        try:
            # Process original query
            query_info = self.query_processor.process_query(query)
            
            # Select enhancement strategy based on query characteristics
            enhanced = self._enhance_based_on_difficulty(query_info)
            
            # Add expansions based on domain
            expansions = self._expand_query(query_info)
            
            return {
                'original_query': query,
                'enhanced_query': enhanced,
                'expansions': expansions,
                'query_info': query_info
            }
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            raise

    def _enhance_based_on_difficulty(self, query_info: Dict) -> str:
        """Select enhancement strategy based on query characteristics"""
        difficulty = query_info['difficulty']['difficulty_level']
        intent = query_info['intent']['primary'][0].value
        
        # Choose strategy based on query characteristics
        if difficulty in ['difficult', 'very_difficult']:
            return self.strategies.enhance_difficulty_based(query_info)
        elif intent in ['technical', 'domain_specific']:
            return self.strategies.enhance_domain_specific(query_info)
        else:
            return self.strategies.enhance_intent_focused(query_info)

    def _expand_query(self, query_info: Dict) -> List[str]:
        """Generate query expansions"""
        prompt = self.templates.QUERY_EXPANSION.format(
            query=query_info['original_query']
        )
        
        expansion_text = self.model.generate(prompt)
        return [term.strip() for term in expansion_text.split('\n') if term.strip()]