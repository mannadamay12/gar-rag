# tests/rag/generator/test_generator.py
import pytest
from src.rag.generator.flan_generator import FlanGenerator

def test_generator_initialization():
    """Test generator initialization"""
    generator = FlanGenerator()
    assert generator.model is not None
    assert generator.tokenizer is not None

def test_context_formatting(sample_documents):
    """Test context formatting"""
    generator = FlanGenerator()
    formatted = generator._format_context(sample_documents)
    
    assert isinstance(formatted, str)
    for doc in sample_documents:
        assert doc.content in formatted

def test_generate_with_context(sample_documents):
    """Test generation with context"""
    generator = FlanGenerator()
    result = generator.generate(
        query="What causes climate change?",
        contexts=sample_documents[:2],
        enhanced_query="What are the primary factors contributing to climate change?"
    )
    
    assert 'answer' in result
    assert 'sources' in result
    assert 'scores' in result
    assert isinstance(result['answer'], str)
    assert len(result['answer']) > 0