from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np
from .document import Document

class DocumentStoreError(Exception):
    """Base exception class for document store errors"""
    pass

class BaseDocumentStore(ABC):
    """Abstract base class for document stores"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 index_name: str = "document_store",
                 **kwargs):
        """Initialize document store"""
        self.embedding_dim = embedding_dim
        self.index_name = index_name
        self._initialize_store(**kwargs)

    @abstractmethod
    def _initialize_store(self, **kwargs) -> None:
        """Initialize underlying store implementation"""
        pass

    @abstractmethod
    def add_documents(self, 
                     documents: List[Document],
                     batch_size: int = 1000) -> bool:
        """Add documents to store"""
        pass

    @abstractmethod
    def get_document_by_id(self, id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        pass

    @abstractmethod
    def search_documents(self,
                        query_embedding: np.ndarray,
                        top_k: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def update_document(self,
                       id: str,
                       document: Document) -> bool:
        """Update existing document"""
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from store"""
        pass

    @abstractmethod
    def get_all_documents(self,
                         filters: Optional[Dict[str, Any]] = None,
                         batch_size: int = 1000) -> List[Document]:
        """Get all documents, optionally filtered"""
        pass

    def update_embeddings(self,
                         documents: List[Document]) -> bool:
        """Update embeddings for existing documents"""
        try:
            return self.add_documents(documents)
        except Exception as e:
            raise DocumentStoreError(f"Failed to update embeddings: {str(e)}")