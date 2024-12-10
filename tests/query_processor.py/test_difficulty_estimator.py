import pytest
from src.query_processor.difficulty_estimator import QueryDifficultyEstimator

@pytest.fixture
def estimator():
    return QueryDifficultyEstimator()

def test_simple_query(estimator):
    query_info = {
        'token_count': 2,
        'token_info': {
            'pos_tags': [('what', 'WP'), ('time', 'NN')]
        },
        'entities': [],
        'ngrams': {
            'trigrams': []
        }
    }
    
    result = estimator.estimate(query_info)
    assert result['difficulty_level'] == 'easy'
    assert result['final_score'] < 0.5

def test_complex_query(estimator):
    query_info = {
        'token_count': 8,
        'token_info': {
            'pos_tags': [
                ('what', 'WP'), ('are', 'VBP'), ('the', 'DT'),
                ('environmental', 'JJ'), ('impacts', 'NNS'),
                ('of', 'IN'), ('renewable', 'JJ'), ('energy', 'NN')
            ]
        },
        'entities': ['renewable energy'],
        'ngrams': {
            'trigrams': ['environmental impacts of', 'impacts of renewable']
        }
    }
    
    result = estimator.estimate(query_info)
    assert result['difficulty_level'] in ['difficult', 'very_difficult']
    assert result['final_score'] > 0.7

def test_edge_cases(estimator):
    # Empty query
    empty_query = {
        'token_count': 0,
        'token_info': {'pos_tags': []},
        'entities': [],
        'ngrams': {'trigrams': []}
    }
    
    result = estimator.estimate(empty_query)
    assert 'final_score' in result
    assert 'difficulty_level' in result