# src/rag/ranking/ranker.py
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from ..document_store.document import Document

class ResultRanker:
    def __init__(self, bm25_weight: float = 0.3):
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """Min-max normalization of scores"""
        scores = np.array(scores)
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        score_range = scores.max() - min_score
        if score_range == 0:
            return np.ones_like(scores)
        return (scores - min_score) / score_range

    def _calculate_bm25_scores(self, query: str, documents: List[Document]) -> List[float]:
        """Calculate BM25 scores for documents"""
        # Tokenize documents
        tokenized_docs = [doc.content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get scores
        query_tokens = query.lower().split()
        return bm25.get_scores(query_tokens)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using combined vector similarity and BM25 scores"""
        if not documents:
            return documents

        # Get vector similarity scores
        vector_scores = self._normalize_scores([doc.score for doc in documents])
        
        # Calculate BM25 scores
        bm25_scores = self._normalize_scores(self._calculate_bm25_scores(query, documents))
        
        # Combine scores
        combined_scores = (
            self.vector_weight * vector_scores + 
            self.bm25_weight * bm25_scores
        )
        
        # Update document scores
        for doc, score in zip(documents, combined_scores):
            doc.score = float(score)
        
        # Sort by combined score
        return sorted(documents, key=lambda x: x.score, reverse=True)