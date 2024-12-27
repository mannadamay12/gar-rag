from typing import Dict, List
from collections import Counter
from ..utils.logger import setup_logger

logger = setup_logger("difficulty_estimator")

class QueryDifficultyEstimator:
    def __init__(self):
        self.difficulty_thresholds = {
            'easy': 0.4,
            'moderate': 0.7,
            'difficult': 0.9
        }

    def estimate(self, query_info: Dict) -> Dict:
        """
        Estimate query difficulty based on multiple factors
        Args:
            query_info: Dictionary containing processed query information
        Returns:
            Dictionary with difficulty scores and analysis
        """
        try:
            # Calculate individual metrics
            length_score = self._get_length_complexity(query_info)
            term_score = self._get_term_complexity(query_info)
            semantic_score = self._get_semantic_complexity(query_info)
            
            # Combine scores with weights
            final_score = (
                length_score * 0.4 +
                term_score * 0.35 +
                semantic_score * 0.25
            )

            difficulty_level = self._get_difficulty_level(final_score)
            
            logger.info(f"Query difficulty estimated: {difficulty_level} ({final_score:.2f})")
            
            return {
                'final_score': final_score,
                'difficulty_level': difficulty_level,
                'metrics': {
                    'length_complexity': length_score,
                    'term_complexity': term_score,
                    'semantic_complexity': semantic_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error estimating query difficulty: {str(e)}")
            raise

    def _get_length_complexity(self, query_info: Dict) -> float:
        """
        Calculate complexity based on query length
        Short = easier, Long = harder
        """
        token_count = query_info.get('token_count', 0)
        
        # Normalize based on typical query lengths
        if token_count <= 2:
            return 0.2  # Very simple
        elif token_count <= 3:
            return 0.3  # Simple
        elif token_count <= 4:
            return 0.5  # Moderate
        elif token_count <= 6:
            return 0.7  # Complex
        else:
            return 1.0  # Very complex
        
    def _get_term_complexity(self, query_info: Dict) -> float:
        """Calculate complexity based on term characteristics"""
        pos_tags = query_info.get('token_info', {}).get('pos_tags', [])
        
        if not pos_tags:
            return 0.5
            
        # Count different types of terms
        pos_counts = Counter(tag for _, tag in pos_tags)
        
        # Simple queries often have just nouns and maybe one verb
        if len(pos_counts) <= 2 and ('NN' in pos_counts or 'NNP' in pos_counts):
            return 0.3  # Lower complexity for simple noun phrases
            
        return min(1.0, len(set(pos_counts.keys())) / len(pos_tags))

    # def _get_term_complexity(self, query_info: Dict) -> float:
    #     """
    #     Calculate complexity based on term characteristics
    #     Consider POS tags, lemmas, etc.
    #     """
    #     pos_tags = query_info.get('token_info', {}).get('pos_tags', [])
        
    #     if not pos_tags:
    #         return 0.5  # Default if no POS info
            
    #     # Count different types of terms
    #     pos_counts = Counter(tag for _, tag in pos_tags)
        
    #     # Calculate complexity based on POS diversity
    #     noun_count = pos_counts.get('NN', 0) + pos_counts.get('NNP', 0)
    #     verb_count = pos_counts.get('VB', 0) + pos_counts.get('VBP', 0)
    #     adj_count = pos_counts.get('JJ', 0)
        
    #     # More diverse POS = more complex
    #     pos_diversity = len(set(pos_counts.keys())) / len(pos_tags)
        
    #     # Normalize and combine factors
    #     return min(1.0, (noun_count * 0.3 + verb_count * 0.3 + 
    #                     adj_count * 0.2 + pos_diversity * 0.2))

    def _get_semantic_complexity(self, query_info: Dict) -> float:
        """
        Calculate complexity based on semantic features
        Consider entities, n-grams, etc.
        """
        base_score = 0.3
        # Check for entities
        entities = query_info.get('entities', [])
        has_entities = len(entities) > 0
        
        # Check for complex phrases using n-grams
        ngrams = query_info.get('ngrams', {})
        trigrams = ngrams.get('trigrams', [])
        has_phrases = len(trigrams) > 0
        
        # Higher score for queries with entities and complex phrases
        base_score = 0.4
        if has_entities:
            base_score += 0.3
        if has_phrases:
            base_score += 0.3
            
        return min(1.0, base_score)

    def _get_difficulty_level(self, score: float) -> str:
        """Convert numerical score to difficulty level"""
        if score <= self.difficulty_thresholds['easy']:
            return 'easy'
        elif score <= self.difficulty_thresholds['moderate']:
            return 'moderate'
        elif score <= self.difficulty_thresholds['difficult']:
            return 'difficult'
        else:
            return 'very_difficult'