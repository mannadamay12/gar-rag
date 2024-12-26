from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from ..document_store.faiss_store import FAISSDocumentStore
from ..document_store.base_store import Document
from ..utils.logger import setup_logger

logger = setup_logger("hybrid_retriever")

class ScoreCombinationMethod:
    RRF = "reciprocal_rank_fusion"
    LINEAR = "linear_combination"
    MAX = "max_score"
    WEIGHTED = "weighted_average"

class HybridRetriever(BaseRetriever):
    """Combines BM25 and Dense Retrieval (FAISS) results"""
    
    def __init__(self,
                 document_store: FAISSDocumentStore,
                 combination_method: str = ScoreCombinationMethod.RRF,
                 weight_sparse: float = 0.5,
                 top_k_sparse: int = 100,
                 top_k_dense: int = 100,
                 rrf_k: int = 60):
        """
        Initialize hybrid retriever
        
        Args:
            document_store: FAISS document store
            combination_method: Method to combine scores
            weight_sparse: Weight for BM25 scores (0 to 1)
            top_k_sparse: Number of results from BM25
            top_k_dense: Number of results from dense retrieval
            rrf_k: Constant for reciprocal rank fusion
        """
        self.document_store = document_store
        self.bm25_retriever = BM25Retriever()
        self.combination_method = combination_method
        self.weight_sparse = weight_sparse
        self.weight_dense = 1 - weight_sparse
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.rrf_k = rrf_k

    def retrieve(self,
                query: str,
                query_embedding: Optional[np.ndarray] = None,
                top_k: int = 10,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using both BM25 and dense retrieval
        
        Args:
            query: Text query
            query_embedding: Dense query vector
            top_k: Number of results to return
            
        Returns:
            Combined and reranked list of Documents
        """
        try:
            # Get results from both retrievers
            sparse_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=self.top_k_sparse,
                document_store=self.document_store
            )
            
            dense_results = self.document_store.search_documents(
                query_embedding=query_embedding,
                top_k=self.top_k_dense
            ) if query_embedding is not None else []

            # Combine results based on selected method
            if self.combination_method == ScoreCombinationMethod.RRF:
                combined_results = self._reciprocal_rank_fusion(
                    sparse_results, dense_results, top_k
                )
            elif self.combination_method == ScoreCombinationMethod.LINEAR:
                combined_results = self._linear_combination(
                    sparse_results, dense_results, top_k
                )
            elif self.combination_method == ScoreCombinationMethod.MAX:
                combined_results = self._max_score(
                    sparse_results, dense_results, top_k
                )
            elif self.combination_method == ScoreCombinationMethod.WEIGHTED:
                combined_results = self._weighted_average(
                    sparse_results, dense_results, top_k
                )
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")

            return combined_results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise

    def score_documents(self,
                       query: str,
                       documents: List[Document],
                       query_embedding: Optional[np.ndarray] = None) -> List[Document]:
        """Score a specific set of documents using both methods"""
        # Get BM25 scores
        bm25_scored = self.bm25_retriever.score_documents(query, documents)
        
        # Get dense scores if embedding provided
        if query_embedding is not None:
            embeddings = [doc.embedding for doc in documents if doc.embedding is not None]
            if embeddings:
                embeddings = np.vstack(embeddings)
                dense_scores = self.document_store.index.search(
                    query_embedding.reshape(1, -1),
                    embeddings.shape[0]
                )[0]
                
                # Add dense scores to documents
                for doc, dense_score in zip(documents, dense_scores):
                    doc.dense_score = float(dense_score)
            
        # Combine scores using weighted average
        for doc in documents:
            sparse_score = doc.score or 0.0
            dense_score = getattr(doc, 'dense_score', 0.0)
            doc.score = (self.weight_sparse * sparse_score + 
                        self.weight_dense * dense_score)
            
        return sorted(documents, key=lambda x: x.score, reverse=True)

    def _reciprocal_rank_fusion(self,
                               sparse_results: List[Document],
                               dense_results: List[Document],
                               top_k: int) -> List[Document]:
        """Combine results using reciprocal rank fusion"""
        # Create rank dictionaries
        doc_ranks = {}
        
        # Process sparse results
        for rank, doc in enumerate(sparse_results, start=1):
            doc_ranks[doc.id] = doc_ranks.get(doc.id, []) + [rank]
            
        # Process dense results
        for rank, doc in enumerate(dense_results, start=1):
            doc_ranks[doc.id] = doc_ranks.get(doc.id, []) + [rank]
            
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id, ranks in doc_ranks.items():
            rrf_scores[doc_id] = sum(1 / (r + self.rrf_k) for r in ranks)
            
        # Sort documents by RRF score
        sorted_docs = sorted(
            list(set(sparse_results + dense_results)),
            key=lambda x: rrf_scores.get(x.id, 0),
            reverse=True
        )
        
        # Update scores
        for doc in sorted_docs:
            doc.score = rrf_scores.get(doc.id, 0)
            
        return sorted_docs[:top_k]

    def _linear_combination(self,
                          sparse_results: List[Document],
                          dense_results: List[Document],
                          top_k: int) -> List[Document]:
        """Combine results using linear combination of scores"""
        # Normalize scores within each result set
        self._normalize_scores(sparse_results)
        self._normalize_scores(dense_results)
        
        # Combine results
        doc_scores = {}
        
        # Process sparse results
        for doc in sparse_results:
            doc_scores[doc.id] = self.weight_sparse * doc.score
            
        # Process dense results
        for doc in dense_results:
            if doc.id in doc_scores:
                doc_scores[doc.id] += self.weight_dense * doc.score
            else:
                doc_scores[doc.id] = self.weight_dense * doc.score
                
        # Create combined list
        combined_docs = list(set(sparse_results + dense_results))
        for doc in combined_docs:
            doc.score = doc_scores.get(doc.id, 0)
            
        return sorted(combined_docs, key=lambda x: x.score, reverse=True)[:top_k]

    def _max_score(self,
                   sparse_results: List[Document],
                   dense_results: List[Document],
                   top_k: int) -> List[Document]:
        """Combine results by taking maximum of normalized scores"""
        # Normalize scores
        self._normalize_scores(sparse_results)
        self._normalize_scores(dense_results)
        
        # Create score dictionary
        doc_scores = {}
        
        # Process all results
        for doc in sparse_results + dense_results:
            doc_scores[doc.id] = max(
                doc_scores.get(doc.id, 0),
                doc.score
            )
            
        # Update scores and sort
        combined_docs = list(set(sparse_results + dense_results))
        for doc in combined_docs:
            doc.score = doc_scores.get(doc.id, 0)
            
        return sorted(combined_docs, key=lambda x: x.score, reverse=True)[:top_k]

    def _weighted_average(self,
                         sparse_results: List[Document],
                         dense_results: List[Document],
                         top_k: int) -> List[Document]:
        """Combine results using weighted average of scores"""
        return self._linear_combination(sparse_results, dense_results, top_k)

    @staticmethod
    def _normalize_scores(documents: List[Document]) -> None:
        """Min-max normalize scores in-place"""
        if not documents:
            return
            
        scores = [doc.score for doc in documents]
        min_score = min(scores)
        score_range = max(scores) - min_score
        
        if score_range > 0:
            for doc in documents:
                doc.score = (doc.score - min_score) / score_range