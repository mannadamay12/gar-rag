from abc import ABC, abstractmethod
from typing import Dict, List
from ..models.base_model import BaseLanguageModel
from .prompt_templates import PromptTemplates
from datetime import datetime

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
        """Enhanced strategy for complex queries"""
        intent = query_info['intent']['primary'][0].value
        entities = [e['text'] for e in query_info.get('entities', [])]
        
        prompt = f"""
        Enhance this complex search query:
        Query: {query_info['original_query']}
        Intent: {intent}
        Entities: {', '.join(entities)}
        
        Instructions:
        1. Keep the original search intent
        2. Include entity variations and synonyms
        3. Add domain-specific terminology
        4. Consider temporal aspects if relevant
        5. Balance specificity and recall
        
        Enhanced query:
        """
        
        enhanced = model.generate(prompt)
        
        # Add query expansion with related concepts
        expansions = self._get_query_expansions(query_info, model)
        
        return {
            'enhanced_query': enhanced,
            'strategy': 'complex_breakdown',
            'expansions': expansions
        }
        
    def _get_query_expansions(self, query_info: Dict, model: BaseLanguageModel) -> List[str]:
        """Get query expansions for complex queries"""
        expansion_prompt = f"""
        Generate 3 alternative ways to search for:
        {query_info['original_query']}
        
        Focus on:
        - Different terminology
        - Related concepts
        - Specific aspects
        
        Alternative searches:
        """
        
        expansions = model.generate(expansion_prompt)
        return [exp.strip() for exp in expansions.split('\n') if exp.strip()]

class TechnicalQueryStrategy(EnhancementStrategy):
    """Strategy for technical/specialized queries"""
    
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        entities = [e['text'] for e in query_info.get('entities', [])]
        domain = self._detect_technical_domain(query_info)
        
        prompt = PromptTemplates.TECHNICAL_QUERY.format(
            query=query_info['original_query'],
            domain=domain
        )
        
        enhanced = model.generate(prompt)
        return {
            'enhanced_query': enhanced,
            'strategy': 'technical_domain',
            'domain': domain
        }
    
    def _detect_technical_domain(self, query_info: Dict) -> str:
        """Detect technical domain of query"""
        # Implement domain detection logic
        domains = {
            'programming': ['code', 'programming', 'software', 'developer'],
            'medical': ['disease', 'treatment', 'patient', 'clinical'],
            'legal': ['law', 'legal', 'court', 'jurisdiction'],
            'scientific': ['research', 'experiment', 'laboratory', 'scientific']
        }
        # Simple implementation - can be enhanced
        query_terms = set(query_info['token_info']['lemmatized_tokens'])
        for domain, terms in domains.items():
            if any(term in query_terms for term in terms):
                return domain
        return 'general_technical'

class TemporalQueryStrategy(EnhancementStrategy):
    """Strategy for time-sensitive queries"""
    
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        temporal_context = self._extract_temporal_context(query_info)
        
        prompt = PromptTemplates.TEMPORAL_QUERY.format(
            query=query_info['original_query'],
            temporal_context=temporal_context
        )
        
        enhanced = model.generate(prompt)
        return {
            'enhanced_query': enhanced,
            'strategy': 'temporal_aware',
            'temporal_context': temporal_context
        }
    
    def _extract_temporal_context(self, query_info: Dict) -> str:
        """Extract temporal context from query"""
        now = datetime.now()
        temporal_terms = {
            'current': now.strftime("%Y-%m-%d"),
            'today': now.strftime("%Y-%m-%d"),
            'this year': str(now.year),
            'latest': f"as of {now.strftime('%B %Y')}"
        }
        query = query_info['original_query'].lower()
        for term, date in temporal_terms.items():
            if term in query:
                return date
        return 'current'

class LocalizedQueryStrategy(EnhancementStrategy):
    """Strategy for location-based queries"""
    
    def enhance(self, query_info: Dict, model: BaseLanguageModel) -> Dict:
        locations = [e['text'] for e in query_info.get('entities', []) 
                    if e.get('label') in ['GPE', 'LOC']]
        
        base_query = query_info['original_query']
        if locations:
            enhanced = f"{base_query} location:{' '.join(locations)}"
        else:
            enhanced = base_query
            
        return {
            'enhanced_query': enhanced,
            'strategy': 'location_aware',
            'locations': locations
        }