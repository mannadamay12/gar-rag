from enum import Enum
import re
from typing import Dict, List, Tuple
import logging

from ..utils.logger import setup_logger

logger = setup_logger("intent_classifier")

class QueryIntent(Enum):
    INFORMATIONAL = "informational"    # "how to", "what is", etc.
    NAVIGATIONAL = "navigational"      # specific website/brand queries
    TRANSACTIONAL = "transactional"    # "buy", "download", etc.
    COMMERCIAL = "commercial"          # product research
    LOCATION = "location"              # location-based queries

class IntentClassifier:
    def __init__(self):
        # Initialize intent patterns
        self.intent_patterns = {
            QueryIntent.INFORMATIONAL: [
                r"^(?:what|who|when|where|why|how|explain|define|tell me about)",
                r"difference between",
                r"meaning of",
            ],
            QueryIntent.NAVIGATIONAL: [
                r"\b(?:website|site|webpage|url|link)\b",
                r"\b(?:login|signin|signup)\b",
                r"\.com\b",
            ],
            QueryIntent.TRANSACTIONAL: [
                r"\b(?:buy|purchase|order|download|shop|price)\b",
                r"\b(?:cheap|cheapest|best price|discount)\b",
            ],
            QueryIntent.COMMERCIAL: [
                r"\b(?:review|compare|vs|versus|best)\b",
                r"\b(?:price|cost|cheap|expensive)\b",
                r"top \d+",
            ],
            QueryIntent.LOCATION: [
                r"\b(?:near|nearby|around|closest)\b",
                r"\b(?:near me|nearby|location|directions|map)\b",
                r"where is",
                r"(?:shops?|stores?|restaurants?|cafes?|places?).*(?:near|nearby|around|closest)",
            ]
        }

    def classify(self, query: str) -> Dict[QueryIntent, float]:
        """
        Classify query intent and return confidence scores for each intent
        """
        logger.info(f"Classifying intent for query: {query}")
        query = query.lower().strip()
        scores = {intent: 0.0 for intent in QueryIntent}
        
        # Check each intent's patterns
        for intent, patterns in self.intent_patterns.items():
            score = self._calculate_intent_score(query, patterns)
            scores[intent] = score

        # Normalize scores
        total_score = sum(scores.values()) or 1.0
        normalized_scores = {
            intent: score/total_score 
            for intent, score in scores.items()
        }

        logger.debug(f"Intent scores: {normalized_scores}")
        return normalized_scores

    def _calculate_intent_score(self, query: str, patterns: List[str]) -> float:
        """Calculate score for a specific intent based on matching patterns"""
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 1.0
        return score

    def get_primary_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Get the primary intent with highest confidence score"""
        scores = self.classify(query)
        primary_intent = max(scores.items(), key=lambda x: x[1])
        return primary_intent