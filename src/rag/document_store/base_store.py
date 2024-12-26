from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Class representing a document or chunk of text"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate document after initialization"""
        if not self.id or not self.content:
            raise ValueError("Document must have both id and content")
        
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array")

class DocumentStoreError(Exception):
    """Base exception class for document store errors"""
    pass

class BaseDocumentStore(ABC):
    """Abstract base class for document stores
    
    This class defines the interface for document storage and retrieval systems.
    Implementations should handle both document storage and vector embeddings
    for similarity search.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 index_name: str = "document_store",
                 **kwargs):
        """Initialize document store
        
        Args:
            embedding_dim: Dimension of document embeddings
            index_name: Name of the index/collection
            **kwargs: Additional implementation-specific parameters
        """
        self.embedding_dim = embedding_dim
        self.index_name = index_name
        self._initialize_store(**kwargs)

    @abstractmethod
    def _initialize_store(self, **kwargs) -> None:
        """Initialize the underlying store implementation"""
        pass

    @abstractmethod
    def add_documents(self, 
                     documents: List[Document],
                     batch_size: int = 1000) -> bool:
        """Add documents to the store
        
        Args:
            documents: List of Document objects to add
            batch_size: Size of batches for adding documents
        
        Returns:
            bool: True if successful
        
        Raises:
            DocumentStoreError: If documents cannot be added
        """
        pass

    @abstractmethod
    def get_document_by_id(self, id: str) -> Optional[Document]:
        """Retrieve a document by its ID
        
        Args:
            id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    def search_documents(self, 
                        query_embedding: np.ndarray,
                        top_k: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents using embedding
        
        Args:
            query_embedding: Query vector
            top_k: Number of documents to retrieve
            filters: Metadata filters to apply
            
        Returns:
            List of Documents sorted by similarity
        """
        pass

    @abstractmethod
    def update_embeddings(self,
                         documents: List[Document]) -> bool:
        """Update embeddings for existing documents
        
        Args:
            documents: List of documents with new embeddings
            
        Returns:
            bool: True if successful
        """
        pass

    def update_document(self, id: str, document: Document) -> bool:
        """Update an existing document
        
        Args:
            id: ID of document to update
            document: New document data
            
        Returns:
            bool: True if successful
            
        Raises:
            DocumentStoreError: If update fails
        """
        try:
            if self.delete_documents([id]):
                document.updated_at = datetime.now()
                return self.add_documents([document])
            return False
        except Exception as e:
            logger.error(f"Error updating document {id}: {str(e)}")
            raise DocumentStoreError(f"Failed to update document: {str(e)}")

    @abstractmethod
    def get_documents_by_filters(self,
                               filters: Dict[str, Any],
                               limit: Optional[int] = None) -> List[Document]:
        """Get documents matching metadata filters
        
        Args:
            filters: Metadata key-value pairs to match
            limit: Maximum number of documents to return
            
        Returns:
            List of matching Documents
        """
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from store
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            bool: True if successful
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get dimension of document embeddings"""
        return self.embedding_dim

    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents in store"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all documents from store
        
        Returns:
            bool: True if successful
        """
        pass

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding dimension
        
        Args:
            embedding: Document embedding vector
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If embedding dimension doesn't match
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"required dimension {self.embedding_dim}"
            )
        return True

    def _validate_documents(self, documents: List[Document]) -> None:
        """Validate a list of documents
        
        Args:
            documents: List of documents to validate
            
        Raises:
            ValueError: If documents are invalid
        """
        for doc in documents:
            if doc.embedding is not None:
                self.validate_embedding(doc.embedding)