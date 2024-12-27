# tests/rag/ranking/test_ranker.py
import pytest
import numpy as np
from src.rag.ranking.ranker import ResultRanker

def test_score_normalization():
    """Test score normalization"""
    ranker = ResultRanker()
    scores = [0.5, 0.7, 0.3, 0.9]
    normalized = ranker._normalize_scores(scores)
    
    assert len(normalized) == len(scores)
    assert np.min(normalized) == 0
    assert np.max(normalized) == 1

def test_bm25_scoring(sample_documents):
    """Test BM25 scoring"""
    ranker = ResultRanker()
    scores = ranker._calculate_bm25_scores(
        "climate change effects",
        sample_documents
    )
    
    assert len(scores) == len(sample_documents)
    assert all(isinstance(score, float) for score in scores)

def test_reranking(sample_documents):
    """Test document reranking"""
    ranker = ResultRanker()
    reranked = ranker.rerank("climate change", sample_documents)
    
    assert len(reranked) == len(sample_documents)
    # Check if sorted by score
    scores = [doc.score for doc in reranked]
    assert scores == sorted(scores, reverse=True)