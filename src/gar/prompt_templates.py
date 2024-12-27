from typing import Dict
class PromptTemplates:
    ENHANCE_QUERY = """
    Given this search query and its analysis, enhance it to improve search results:
    
    QUERY: {query}
    INTENT: {intent}
    DIFFICULTY: {difficulty}
    ENTITIES: {entities}
    
    Enhance the query by:
    1. Maintaining the original search intent
    2. Adding relevant context and synonyms
    3. Including domain-specific terminology
    4. Considering temporal relevance
    5. Balancing precision and recall
    
    Enhanced query:
    """

    QUERY_EXPANSION = """
    Expand this search query with relevant terms:
    QUERY: {query}
    
    Consider including:
    1. Synonyms and alternate phrasings
    2. Related concepts and terminology
    3. Domain-specific vocabulary
    4. Common co-occurring terms
    5. Relevant attributes or properties

    Provide expanded terms:
    """

    TECHNICAL_QUERY = """
    Enhance this technical search query:
    QUERY: {query}
    DOMAIN: {domain}
    
    Focus on:
    1. Technical terminology
    2. Industry-standard terms
    3. Related technical concepts
    4. Specific methodologies
    5. Relevant tools or technologies

    Enhanced technical query:
    """

    TEMPORAL_QUERY = """
    Enhance this time-sensitive query:
    QUERY: {query}
    TEMPORAL_CONTEXT: {temporal_context}
    
    Consider:
    1. Current relevance
    2. Historical context
    3. Time-based variations
    4. Recent developments
    5. Future implications

    Enhanced temporal query:
    """