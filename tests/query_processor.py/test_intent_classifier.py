import pytest
from src.query_processor.intent_classifier import IntentClassifier, QueryIntent

@pytest.fixture
def classifier():
    return IntentClassifier()

def test_informational_intent(classifier):
    query = "how to make pizza at home"
    result = classifier.get_primary_intent(query)
    assert result[0] == QueryIntent.INFORMATIONAL

def test_navigational_intent(classifier):
    query = "facebook login page"
    result = classifier.get_primary_intent(query)
    assert result[0] == QueryIntent.NAVIGATIONAL

def test_transactional_intent(classifier):
    query = "buy iphone 13 pro max"
    result = classifier.get_primary_intent(query)
    assert result[0] == QueryIntent.TRANSACTIONAL

def test_commercial_intent(classifier):
    query = "best laptops under 1000"
    result = classifier.get_primary_intent(query)
    assert result[0] == QueryIntent.COMMERCIAL

def test_location_intent(classifier):
    query = "coffee shops near me"
    result = classifier.get_primary_intent(query)
    assert result[0] == QueryIntent.LOCATION

def test_multiple_intents(classifier):
    query = "where to buy pizza near me"
    scores = classifier.classify(query)
    # Should have non-zero scores for multiple intents
    assert sum(score > 0 for score in scores.values()) > 1