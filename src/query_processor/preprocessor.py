from typing import Dict
from .tokenizer import QueryTokenizer
from .entity_recognizer import EntityRecognizer
from .intent_classifier import IntentClassifier
from .difficulty_estimator import QueryDifficultyEstimator
from ..utils.logger import setup_logger

logger = setup_logger("query_processor")

class QueryProcessor:
    def __init__(self):
        self.tokenizer = QueryTokenizer()
        self.entity_recognizer = EntityRecognizer()
        self.intent_classifier = IntentClassifier()
        self.difficulty_estimator = QueryDifficultyEstimator()

    def process_query(self, query: str) -> Dict:
        """
        Complete query processing pipeline
        """
        logger.info(f"Processing query: {query}")
        try:
            # Basic processing
            token_info = self.tokenizer.tokenize(query)
            
            # Entity recognition
            entities = self.entity_recognizer.extract_entities(query)
            
            # Intent classification
            intent_info = self.intent_classifier.classify(query)
            primary_intent = self.intent_classifier.get_primary_intent(query)
            
            # Prepare query info for difficulty estimation
            query_info = {
                'original_query': query,
                'token_info': token_info,
                'token_count': len(token_info['word_tokens']),
                'entities': entities,
                'ngrams': {
                    'bigrams': self.tokenizer.get_ngrams(token_info['word_tokens'], 2),
                    'trigrams': self.tokenizer.get_ngrams(token_info['word_tokens'], 3)
                }
            }
            
            # Estimate difficulty
            difficulty_info = self.difficulty_estimator.estimate(query_info)
            
            # Combine all information
            result = {
                **query_info,
                'intent': {
                    'scores': intent_info,
                    'primary': primary_intent
                },
                'difficulty': difficulty_info
            }
            
            logger.info(f"Successfully processed query with difficulty: {difficulty_info['difficulty_level']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise