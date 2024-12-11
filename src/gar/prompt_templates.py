from typing import Dict

class PromptTemplates:
    ENHANCE_QUERY = """
    Given the following search query, enhance it to improve search results:
    Query: {query}
    Intent: {intent}
    Difficulty: {difficulty}
    Entities: {entities}

    Generate an enhanced version that:
    1. Maintains the original intent
    2. Adds relevant context
    3. Includes important related terms

    Enhanced query:
    """

    QUERY_EXPANSION = """
    Expand the following query with relevant terms:
    {query}

    Include terms related to:
    - Synonyms
    - Related concepts
    - Domain-specific terminology

    Expanded terms:
    """