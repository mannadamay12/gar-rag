# tests/test_integration.py
import pytest
from src.pipeline import SearchPipeline

@pytest.mark.integration
def test_complete_search_pipeline(sample_documents):
    """Test complete search pipeline with GAR and RAG"""
    pipeline = SearchPipeline()
    pipeline.rag_pipeline.document_store.add_documents(sample_documents)
    
    result = pipeline.process_query(
        "What are the effects of climate change?"
    )
    
    assert 'answer' in result
    assert 'sources' in result
    assert 'metadata' in result
    assert 'query_enhancement' in result['metadata']