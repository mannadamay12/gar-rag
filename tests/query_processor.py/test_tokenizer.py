import pytest
from src.query_processor.tokenizer import QueryTokenizer

@pytest.fixture
def tokenizer():
    return QueryTokenizer()

def test_tokenization(tokenizer):
    text = "Running quickly through the park"
    result = tokenizer.tokenize(text)
    
    assert 'word_tokens' in result
    assert 'pos_tags' in result
    assert 'lemmatized_tokens' in result
    assert 'sentence_tokens' in result
    
    # Check lemmatization
    assert 'running' in result['word_tokens']
    assert 'run' in result['lemmatized_tokens']

def test_ngrams(tokenizer):
    tokens = ['this', 'is', 'a', 'test']
    bigrams = tokenizer.get_ngrams(tokens, 2)
    
    assert len(bigrams) == 3
    assert 'this is' in bigrams