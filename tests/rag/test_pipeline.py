# tests/rag/test_pipeline.py
import pytest
from src.rag.pipeline import RAGPipeline

def test_pipeline_initialization():
    """Test pipeline initialization"""
    pipeline = RAGPipeline()
    assert pipeline.embedder is not None
    assert pipeline.document_store is not None
    assert pipeline.generator is not None
    assert pipeline.ranker is not None

def test_document_retrieval(rag_pipeline, sample_documents):
    """Test document retrieval"""
    # Add documents to store
    rag_pipeline.document_store.add_documents(sample_documents)
    
    # Test retrieval
    docs = rag_pipeline._retrieve("climate change")
    assert len(docs) > 0
    assert all(isinstance(doc.score, float) for doc in docs)

def test_query_processing(rag_pipeline, sample_documents):
    """Test complete query processing"""
    # Add documents to store
    rag_pipeline.document_store.add_documents(sample_documents)
    
    # Process query
    result = rag_pipeline.process(
        query="What causes climate change?",
        enhanced_query="What are the primary factors contributing to climate change and global warming?"
    )
    
    assert 'answer' in result
    assert 'sources' in result
    assert 'metadata' in result
    assert result['metadata']['num_docs'] > 0

def test_empty_retrieval_handling(rag_pipeline):
    """Test handling of queries with no relevant documents"""
    result = rag_pipeline.process("completely unrelated query")
    
    assert result['answer'] == "No relevant information found."
    assert len(result['sources']) == 0
    assert result['metadata']['num_docs'] == 0