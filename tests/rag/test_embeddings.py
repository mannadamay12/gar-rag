# tests/rag/test_embeddings.py
import pytest
import numpy as np
from src.rag.embeddings.sentence_embedder import SentenceEmbedder

@pytest.fixture
def embedder():
    return SentenceEmbedder()

def test_embedding_generation():
    """Test basic embedding generation"""
    embedder = SentenceEmbedder()
    text = "This is a test sentence"
    embedding = embedder.embed(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # For MiniLM-L6

def test_batch_embedding():
    """Test batch embedding"""
    embedder = SentenceEmbedder()
    texts = [
        "First sentence",
        "Second sentence",
        "Third sentence"
    ]
    embeddings = embedder.embed_batch(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)
    assert np.all(np.isfinite(embeddings))

def test_embedding_similarity():
    """Test embedding similarity"""
    embedder = SentenceEmbedder()
    
    # Similar sentences should have higher similarity
    sent1 = "The cat sat on the mat"
    sent2 = "A cat is sitting on a mat"
    sent3 = "Quantum physics is complex"
    
    emb1 = embedder.embed(sent1)
    emb2 = embedder.embed(sent2)
    emb3 = embedder.embed(sent3)
    
    sim12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    assert sim12 > sim13  # Similar sentences have higher similarity