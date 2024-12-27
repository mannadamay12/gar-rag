# tests/gar/test_enhancer.py
import pytest
from unittest.mock import Mock, patch
from src.gar.enhancer import GAREnhancer
from src.models.flan_t5 import FlanT5Model

@pytest.fixture
def mock_flan_t5():
    class MockModel:
        def generate(self, prompt):
            return f"Enhanced: {prompt.split('Query:')[1].strip()}"
    
    return MockModel()

@pytest.fixture
def enhancer(mock_flan_t5):
    """Create enhancer with mock model"""
    with patch('src.gar.enhancer.T5ForConditionalGeneration') as mock_model:
        with patch('src.gar.enhancer.T5Tokenizer') as mock_tokenizer:
            mock_model.return_value.generate.return_value = [0]  # Dummy token ID
            mock_tokenizer.return_value.decode.return_value = "Enhanced query result"
            enhancer = GAREnhancer(model_type="flan-t5")
            # Replace the model with our mock
            enhancer.model = mock_flan_t5
            return enhancer

def test_query_enhancement_basic(enhancer):
    """Test basic query enhancement using Flan-T5"""
    query = "climate change effects"
    result = enhancer.enhance_query(query)
    
    assert result['original_query'] == query
    assert isinstance(result['enhanced_query'], str)
    assert len(result['enhanced_query']) > len(query)
    assert 'strategy_used' in result
    assert 'query_info' in result

def test_query_enhancement_strategies(enhancer):
    """Test different enhancement strategies with Flan-T5"""
    queries = {
        "weather forecast": "simple_expansion",
        "quantum computing applications": "technical_domain",
        "environmental impact of renewable energy in developing economies": "complex_breakdown"
    }
    
    for query, expected_strategy in queries.items():
        result = enhancer.enhance_query(query)
        assert result['strategy_used'] == expected_strategy

def test_enhancement_quality(enhancer):
    """Test quality of Flan-T5 enhancements"""
    original = "electric cars environmental impact"
    enhanced = enhancer.enhance_query(original)
    
    assert len(enhanced['enhanced_query']) > len(original)
    assert any(term in enhanced['enhanced_query'].lower() 
              for term in ['electric', 'environmental', 'impact'])