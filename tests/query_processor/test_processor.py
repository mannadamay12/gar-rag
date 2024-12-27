# tests/query_processor/test_processor.py
import pytest
from src.query_processor.preprocessor import QueryProcessor

@pytest.fixture
def processor():
    return QueryProcessor()

def test_query_processing_basic():
    """Test basic query processing"""
    processor = QueryProcessor()
    query = "What is the impact of climate change?"
    result = processor.process_query(query)
    
    assert 'original_query' in result
    assert 'token_info' in result
    assert 'entities' in result
    assert 'intent' in result
    assert 'difficulty' in result

def test_query_processing_difficulty():
    """Test difficulty estimation"""
    processor = QueryProcessor()
    # Simple query
    simple_query = "weather today"
    simple_result = processor.process_query(simple_query)
    assert simple_result['difficulty']['difficulty_level'] == 'easy'
    
    # Complex query
    complex_query = "What are the environmental and economic implications of renewable energy adoption in developing countries?"
    complex_result = processor.process_query(complex_query)
    assert complex_result['difficulty']['difficulty_level'] in ['difficult', 'very_difficult']

def test_query_processing_entities():
    """Test entity recognition"""
    processor = QueryProcessor()
    query = "How does Tesla impact climate change in California?"
    result = processor.process_query(query)
    
    entities = [e['text'] for e in result['entities']]
    assert 'Tesla' in entities
    assert 'California' in entities

def test_query_processing_intent():
    """Test intent classification"""
    processor = QueryProcessor()
    queries = {
        "how to make pizza": "informational",
        "buy iphone 13": "transactional",
        "facebook login": "navigational"
    }
    
    for query, expected_intent in queries.items():
        result = processor.process_query(query)
        assert result['intent']['primary'][0].value == expected_intent