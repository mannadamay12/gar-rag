from typing import Dict, List, Type
from functools import lru_cache
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ..query_processor.preprocessor import QueryProcessor
from ..query_processor.tokenizer import QueryTokenizer
from ..utils.logger import setup_logger
from .strategies import (
    EnhancementStrategy,
    SimpleExpansionStrategy,
    DomainSpecificStrategy,
    ComplexQueryStrategy,
    TechnicalQueryStrategy,
    TemporalQueryStrategy,
    LocalizedQueryStrategy
)

logger = setup_logger("gar_enhancer")

class GAREnhancer:
    def __init__(self, model_type: str = "flan-t5"):
        self.query_processor = QueryProcessor()
        self.tokenizer = QueryTokenizer()
        self.model = self._get_model(model_type)
        self.enhancement_cache = {}
        self.strategies = {
            'easy': SimpleExpansionStrategy(),
            'moderate': DomainSpecificStrategy(),
            'difficult': ComplexQueryStrategy(),
            'very_difficult': ComplexQueryStrategy(),
            'technical': TechnicalQueryStrategy(),
            'temporal': TemporalQueryStrategy(),
            'localized': LocalizedQueryStrategy()
        }

    def _get_model(self, model_type: str):
        """Initialize the T5 model"""
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        class Model:
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model
                
            def generate(self, prompt):
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                outputs = self.model.generate(**inputs, max_length=512)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return Model(tokenizer, model)


    def _select_strategy(self, query_info: Dict) -> EnhancementStrategy:
        """Select appropriate strategy based on query analysis"""
        # Check for technical terms
        if self._is_technical_query(query_info):
            return self.strategies['technical']
            
        # Check for temporal indicators
        if self._has_temporal_indicators(query_info):
            return self.strategies['temporal']
            
        # Check for location entities
        if self._has_location_entities(query_info):
            return self.strategies['localized']
            
        # Fall back to difficulty-based selection
        return self.strategies[query_info['difficulty']['difficulty_level']]
    
    @lru_cache(maxsize=1000)    
    def enhance_query(self, query: str) -> Dict:
        """Enhanced query enhancement with caching and error handling"""
        try:
            # Process query
            query_info = self.query_processor.process_query(query)
            
            # Select enhancement strategy
            difficulty = query_info['difficulty']['difficulty_level']
            strategy = self.strategies[difficulty]
            
            # Enhance query with fallback handling
            try:
                enhanced = strategy.enhance(query_info, self.model)
            except Exception as e:
                logger.warning(f"Primary strategy failed, falling back to simple expansion: {e}")
                enhanced = self.strategies['easy'].enhance(query_info, self.model)
            
            # Validate enhancement
            if not self._validate_enhancement(enhanced['enhanced_query'], query):
                logger.warning("Enhancement validation failed, using original query")
                enhanced['enhanced_query'] = query
                
            return {
                'original_query': query,
                'enhanced_query': enhanced['enhanced_query'],
                'strategy_used': enhanced['strategy'],
                'query_info': query_info
            }
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            raise

    def _validate_enhancement(self, enhanced_query: str, original_query: str) -> bool:
        """Validate the enhanced query"""
        if not enhanced_query or len(enhanced_query.split()) < 2:
            return False
            
        # Check if enhancement maintains core meaning
        original_tokens = set(self.tokenizer.tokenize(original_query)['word_tokens'])
        enhanced_tokens = set(self.tokenizer.tokenize(enhanced_query)['word_tokens'])
        
        # At least 50% of original terms should be present
        overlap = len(original_tokens.intersection(enhanced_tokens))
        return overlap >= len(original_tokens) * 0.5