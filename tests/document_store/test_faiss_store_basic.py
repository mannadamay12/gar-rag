import pytest
import numpy as np
from src.rag.document_store.base_store import Document, DocumentStoreError

def test_add_documents(store, encoder):
    # Create test documents
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy fox and dog are sleeping"
    ]
    embeddings = encoder.encode(texts)
    docs = [
        Document(
            id=f"doc_{i}",
            content=text,
            embedding=embedding,
            metadata={"source": "test"}
        ) for i, (text, embedding) in enumerate(zip(texts, embeddings))
    ]
    
    assert store.add_documents(docs)
    assert store.get_document_count() == 3

def test_search_documents(store, encoder):
    query = "quick brown fox"
    query_embedding = encoder.encode(query)
    
    results = store.search_documents(query_embedding, top_k=2)
    assert len(results) == 2
    assert results[0].score > results[1].score

def test_filter_search(store, encoder):
    query = "lazy animals"
    query_embedding = encoder.encode(query)
    
    results = store.search_documents(
        query_embedding,
        filters={"source": "test"},
        top_k=2
    )
    assert all(doc.metadata["source"] == "test" for doc in results)

def test_update_delete_documents(store):
    # Test update
    doc_id = "doc_0"
    new_doc = Document(
        id=doc_id,
        content="Updated content",
        embedding=np.random.rand(384),
        metadata={"source": "updated"}
    )
    assert store.update_document(doc_id, new_doc)
    
    # Test delete
    assert store.delete_documents([doc_id])
    assert store.get_document_by_id(doc_id) is None