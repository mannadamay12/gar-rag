import pytest
from tqdm import tqdm
from src.rag.document_store.base_store import Document, DocumentStoreError

def test_msmarco_indexing(store, encoder, msmarco_samples):
    # Prepare documents
    docs = []
    for idx, sample in enumerate(tqdm(msmarco_samples)):
        embedding = encoder.encode(sample['query'])
        doc = Document(
            id=f"msmarco_{idx}",
            content=sample['query'],
            embedding=embedding,
            metadata={
                "answers": sample['answers'],
                "source": "msmarco"
            }
        )
        docs.append(doc)
    
    # Test batch indexing
    assert store.add_documents(docs, batch_size=100)
    assert store.get_document_count() == len(docs)

def test_msmarco_retrieval(store, encoder, msmarco_samples):
    # Test retrieval quality
    query = msmarco_samples[0]['query']
    query_embedding = encoder.encode(query)
    
    results = store.search_documents(query_embedding, top_k=5)
    
    # Check if top result matches the query
    assert results[0].content == query
    assert results[0].score > 0.8  # High similarity expected for exact match

def test_msmarco_batch_search(store, encoder, msmarco_samples):
    # Test batch search performance
    queries = [sample['query'] for sample in msmarco_samples[:10]]
    query_embeddings = encoder.encode(queries)
    
    for query_embedding in query_embeddings:
        results = store.search_documents(query_embedding, top_k=3)
        assert len(results) == 3