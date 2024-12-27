# tests/rag/test_document_store.py
import pytest
import numpy as np
from src.rag.document_store.faiss_store import FAISSDocumentStore
from src.rag.document_store.document import Document

@pytest.fixture
def doc_store():
    return FAISSDocumentStore(embedding_dim=768)

@pytest.fixture
def sample_docs():
    docs = [
        Document(
            id="1",
            content="Climate change effects on environment",
            embedding=np.random.rand(768),
            metadata={"source": "research"}
        ),
        Document(
            id="2",
            content="Renewable energy development",
            embedding=np.random.rand(768),
            metadata={"source": "news"}
        )
    ]
    return docs

def test_document_store_basic(doc_store, sample_docs):
    """Test basic document store operations"""
    # Add documents
    assert doc_store.add_documents(sample_docs)
    
    # Retrieve document
    doc = doc_store.get_document_by_id("1")
    assert doc.content == "Climate change effects on environment"
    
    # Search documents
    query_vector = np.random.rand(768)
    results = doc_store.search_documents(query_vector, top_k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

def test_document_store_filters(doc_store, sample_docs):
    """Test document filtering"""
    doc_store.add_documents(sample_docs)
    
    # Search with filters
    query_vector = np.random.rand(768)
    results = doc_store.search_documents(
        query_vector,
        filters={"source": "research"},
        top_k=2
    )
    assert len(results) == 1
    assert results[0].metadata["source"] == "research"

def test_document_updates(doc_store, sample_docs):
    """Test document updates"""
    doc_store.add_documents(sample_docs)
    
    # Update document
    updated_doc = Document(
        id="1",
        content="Updated content",
        embedding=np.random.rand(768),
        metadata={"source": "updated"}
    )
    assert doc_store.update_document("1", updated_doc)
    
    # Verify update
    doc = doc_store.get_document_by_id("1")
    assert doc.content == "Updated content"
    assert doc.metadata["source"] == "updated"