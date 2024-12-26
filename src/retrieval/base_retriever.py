from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from ..document_store.base_store import Document

class BaseRetriever(ABC):
    """Abstract base class for retrieval methods"""
    
    @abstractmethod
    def retrieve(self, 
                query: str,
                top_k: int = 10,
                **kwargs) -> List[Document]:
        """Retrieve relevant documents for query"""
        pass
    
    @abstractmethod
    def score_documents(self,
                       query: str,
                       documents: List[Document]) -> List[Document]:
        """Score a list of documents against query"""
        pass