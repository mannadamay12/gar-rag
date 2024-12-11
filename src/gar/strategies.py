# src/gar/strategies.py
from abc import ABC, abstractmethod
from typing import Dict, List
from ..models.base_model import BaseLanguageModel

class EnhancementStrategy(ABC):
    @abstractmethod
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        """Enhance query using specific strategy"""
        pass

class SimpleExpansionStrategy(EnhancementStrategy):
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        """For simple queries - expand with synonyms and related terms"""
        prompt = f"""
        Enhance this simple search query with related terms:
        Query: {query_info['original_query']}
        
        Provide:
        1. Synonyms
        2. Related terms
        3. Common variations
        """
        enhanced = model.generate(prompt)
        return {
            'enhanced_query': enhanced,
            'strategy': 'simple_expansion'
        }

class DomainSpecificStrategy(EnhancementStrategy):
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        """For domain-specific queries - add technical terms"""
        entities = [e['text'] for e in query_info.get('entities', [])]
        prompt = f"""
        Enhance this domain-specific query with technical terms:
        Query: {query_info['original_query']}
        Entities: {', '.join(entities)}
        
        Include:
        1. Technical terminology
        2. Domain-specific concepts
        3. Related technical terms
        """
        enhanced = model.generate(prompt)
        return {
            'enhanced_query': enhanced,
            'strategy': 'domain_specific'
        }

class ComplexQueryStrategy(EnhancementStrategy):
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        """For complex queries - break down and expand"""
        intent = query_info['intent']['primary'][0].value
        prompt = f"""
        Break down and enhance this complex query:
        Query: {query_info['original_query']}
        Intent: {intent}
        
        Provide:
        1. Main concepts
        2. Related aspects
        3. Expanded search terms
        """
        enhanced = model.generate(prompt)
        return {
            'enhanced_query': enhanced,
            'strategy': 'complex_breakdown'
        }