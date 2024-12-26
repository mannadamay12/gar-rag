import pytest
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score
from src.rag.document_store.base_store import Document, DocumentStoreError

def test_beir_indexing(store, encoder, beir_samples):
    # Prepare documents
    docs = []
    for idx, sample in enumerate(tqdm(beir_samples)):
        text = sample['text']
        embedding = encoder.encode(text)
        doc = Document(
            id=f"beir_{idx}",
            content=text,
            embedding=embedding,
            metadata={
                "title": sample.get('title', ''),
                "source": "beir"
            }
        )
        docs.append(doc)
    
    assert store.add_documents(docs, batch_size=100)

def test_beir_retrieval_quality(store, encoder, beir_samples):
    # Test retrieval quality using BEIR queries
    retrieved_correct = 0
    total_queries = min(100, len(beir_samples))
    
    for idx in range(total_queries):
        query = beir_samples[idx]['text']
        query_embedding = encoder.encode(query)
        
        results = store.search_documents(
            query_embedding,
            top_k=10,
            filters={"source": "beir"}
        )
        
        # Check if relevant documents are retrieved
        relevant_docs = set(beir_samples[idx].get('relevant_docs', []))
        retrieved_docs = set(doc.id for doc in results)
        
        if relevant_docs & retrieved_docs:
            retrieved_correct += 1
    
    accuracy = retrieved_correct / total_queries
    assert accuracy > 0.3  # Expect at least 30% accuracy

def test_beir_filtering(store, encoder, beir_samples):
    # Test metadata filtering
    query = beir_samples[0]['text']
    query_embedding = encoder.encode(query)
    
    # Search with title filter
    title = beir_samples[0]['title']
    results = store.search_documents(
        query_embedding,
        filters={"title": title},
        top_k=5
    )
    
    assert all(doc.metadata["title"] == title for doc in results)