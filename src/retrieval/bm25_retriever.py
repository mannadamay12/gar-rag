import numpy as np
from typing import List, Dict, Optional
import math
from collections import Counter
from .base_retriever import BaseRetriever
from ..document_store.base_store import Document
from ..query_processor.preprocessor import QueryProcessor

class BM25Retriever(BaseRetriever):
    """BM25 implementation for document retrieval"""
    
    def __init__(self, 
                 k1: float = 1.5,
                 b: float = 0.75,
                 epsilon: float = 0.25):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Smoothing parameter
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.query_processor = QueryProcessor()
        
        # Initialize index statistics
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.doc_freqs: Dict[str, Dict[str, int]] = {}  # term -> {doc_id -> freq}
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0
        
    def index_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from documents"""
        # Reset index statistics
        self.doc_lengths.clear()
        self.doc_freqs.clear()
        self.idf.clear()
        
        # Calculate document lengths and term frequencies
        total_length = 0
        for doc in documents:
            # Tokenize document content
            tokens = self._tokenize(doc.content)
            doc_length = len(tokens)
            
            # Store document length
            self.doc_lengths[doc.id] = doc_length
            total_length += doc_length
            
            # Calculate term frequencies
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = {}
                self.doc_freqs[term][doc.id] = freq
        
        self.total_docs = len(documents)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        # Calculate IDF scores
        self._calculate_idf()
    
    def retrieve(self,
                query: str,
                top_k: int = 10,
                **kwargs) -> List[Document]:
        """
        Retrieve top-k documents for query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of Documents with scores
        """
        # Process query
        query_terms = self._tokenize(query)
        
        # Calculate scores for all documents
        scores: Dict[str, float] = {}
        for term in query_terms:
            if term not in self.doc_freqs:
                continue
                
            idf = self.idf.get(term, 0)
            
            for doc_id, term_freq in self.doc_freqs[term].items():
                if doc_id not in scores:
                    scores[doc_id] = 0
                    
                # Calculate BM25 score for term-document pair
                doc_length = self.doc_lengths[doc_id]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                scores[doc_id] += idf * (numerator / denominator)
        
        # Sort documents by score
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return top-k documents
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = kwargs.get('document_store').get_document_by_id(doc_id)
            if doc:
                doc.score = scores[doc_id]
                results.append(doc)
                
        return results
    
    def score_documents(self,
                       query: str,
                       documents: List[Document]) -> List[Document]:
        """Score a specific set of documents"""
        query_terms = self._tokenize(query)
        scored_docs = []
        
        for doc in documents:
            score = 0
            doc_terms = self._tokenize(doc.content)
            term_freqs = Counter(doc_terms)
            
            for term in query_terms:
                if term not in term_freqs:
                    continue
                    
                idf = self.idf.get(term, 0)
                term_freq = term_freqs[term]
                doc_length = len(doc_terms)
                
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * (numerator / denominator)
            
            doc.score = score
            scored_docs.append(doc)
            
        return sorted(scored_docs, key=lambda x: x.score, reverse=True)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using query processor"""
        processed = self.query_processor.process_query(text)
        return processed['filtered_tokens']
    
    def _calculate_idf(self) -> None:
        """Calculate IDF scores for all terms"""
        for term, doc_freqs in self.doc_freqs.items():
            # Number of documents containing the term
            doc_count = len(doc_freqs)
            # Calculate IDF with smoothing
            self.idf[term] = math.log(1 + (self.total_docs - doc_count + 0.5)/(doc_count + 0.5))