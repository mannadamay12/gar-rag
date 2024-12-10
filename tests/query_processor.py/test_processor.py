import pytest
from src.query_processor.preprocessor import QueryPreprocessor

@pytest.fixture
def preprocessor():
    return QueryPreprocessor()

def test_clean_query(preprocessor):
    query = "What's the Impact of AI?!"
    cleaned = preprocessor.clean_query(query)
    assert cleaned == "whats the impact of ai"

def test_tokenize(preprocessor):
    query = "machine learning basics"
    tokens = preprocessor.tokenize(query)
    assert tokens == ['machine', 'learning', 'basics']

def test_remove_stopwords(preprocessor):
    tokens = ['what', 'is', 'the', 'impact', 'of', 'ai']
    filtered = preprocessor.remove_stopwords(tokens)
    assert filtered == ['impact', 'ai']

def test_process_query(preprocessor):
    query = "What's the impact of AI?!"
    result = preprocessor.process_query(query)
    assert 'original_query' in result
    assert 'cleaned_query' in result
    assert 'filtered_tokens' in result
    assert 'token_count' in result