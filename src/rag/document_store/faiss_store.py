import faiss
import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict
import logging
from datetime import datetime
from .base_store import BaseDocumentStore, Document, DocumentStoreError

logger = logging.getLogger(__name__)

class FAISSDocumentStore(BaseDocumentStore):
    """FAISS-based document store implementation"""
    
    def __init__(self,
                 embedding_dim: int = 768,
                 index_name: str = "document_store",
                 similarity: str = "cosine",
                 **kwargs):
        """
        Initialize FAISS document store
        
        Args:
            embedding_dim: Dimension of embeddings
            index_name: Name of the index
            similarity: 'cosine' or 'l2' distance
        """
        self.similarity = similarity
        super().__init__(embedding_dim=embedding_dim, index_name=index_name, **kwargs)
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        
        # FAISS ID mapping
        self.doc_to_faiss_id: Dict[str, int] = {}
        self.faiss_to_doc_id: Dict[int, str] = {}
        self.next_faiss_id = 0

    def _initialize_store(self, **kwargs) -> None:
        """Initialize FAISS index"""
        if self.similarity == "cosine":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_documents(self,
                     documents: List[Document],
                     batch_size: int = 1000) -> bool:
        """Add documents to store"""
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                embeddings_batch = []
                
                for doc in batch:
                    if doc.id in self.documents:
                        logger.warning(f"Document {doc.id} already exists, updating...")
                        self.update_document(doc.id, doc)
                        continue
                        
                    if doc.embedding is not None:
                        # Store document
                        self.documents[doc.id] = doc
                        self.doc_to_faiss_id[doc.id] = self.next_faiss_id
                        self.faiss_to_doc_id[self.next_faiss_id] = doc.id
                        
                        # Prepare embedding
                        if self.similarity == "cosine":
                            embedding = doc.embedding / np.linalg.norm(doc.embedding)
                        else:
                            embedding = doc.embedding
                        embeddings_batch.append(embedding)
                        
                        self.next_faiss_id += 1
                
                if embeddings_batch:
                    self.index.add(np.array(embeddings_batch))
            
            return True
            
        except Exception as e:
            raise DocumentStoreError(f"Error adding documents: {str(e)}")

    def search_documents(self,
                        query_embedding: np.ndarray,
                        top_k: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if self.similarity == "cosine":
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search FAISS index
            scores, faiss_ids = self.index.search(
                query_embedding.reshape(1, -1), 
                top_k
            )
            
            # Convert results to documents
            results = []
            for score, faiss_id in zip(scores[0], faiss_ids[0]):
                if faiss_id == -1:
                    continue
                    
                doc_id = self.faiss_to_doc_id[faiss_id]
                doc = self.documents[doc_id]
                
                if filters and not self._check_filters(doc, filters):
                    continue
                    
                doc.score = float(score)
                results.append(doc)
            
            return results
            
        except Exception as e:
            raise DocumentStoreError(f"Error during search: {str(e)}")

    def _check_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches filters"""
        for key, value in filters.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False
        return True

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from store"""
        try:
            # Remove from FAISS
            faiss_ids_to_remove = []
            for doc_id in ids:
                if doc_id in self.doc_to_faiss_id:
                    faiss_ids_to_remove.append(self.doc_to_faiss_id[doc_id])
                    del self.documents[doc_id]
                    del self.doc_to_faiss_id[doc_id]
            
            if faiss_ids_to_remove:
                self._rebuild_index_without_ids(faiss_ids_to_remove)
            return True
            
        except Exception as e:
            raise DocumentStoreError(f"Error deleting documents: {str(e)}")

    def _rebuild_index_without_ids(self, faiss_ids_to_remove: List[int]) -> None:
        """Rebuild FAISS index excluding specified IDs"""
        # Get remaining embeddings
        remaining_embeddings = []
        new_faiss_to_doc_id = {}
        new_doc_to_faiss_id = {}
        new_faiss_id = 0
        
        for old_faiss_id, doc_id in self.faiss_to_doc_id.items():
            if old_faiss_id not in faiss_ids_to_remove:
                doc = self.documents[doc_id]
                embedding = doc.embedding
                
                if self.similarity == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)
                remaining_embeddings.append(embedding)
                
                new_faiss_to_doc_id[new_faiss_id] = doc_id
                new_doc_to_faiss_id[doc_id] = new_faiss_id
                new_faiss_id += 1
        
        # Rebuild index
        self._initialize_store()
        if remaining_embeddings:
            self.index.add(np.array(remaining_embeddings))
        
        # Update mappings
        self.faiss_to_doc_id = new_faiss_to_doc_id
        self.doc_to_faiss_id = new_doc_to_faiss_id
        self.next_faiss_id = new_faiss_id

    def get_document_by_id(self, id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(id)

    def get_all_documents(self,
                         filters: Optional[Dict[str, Any]] = None,
                         batch_size: int = 1000) -> List[Document]:
        """Get all documents, optionally filtered"""
        docs = []
        for doc in self.documents.values():
            if not filters or self._check_filters(doc, filters):
                docs.append(doc)
        return docs

    def update_document(self, id: str, document: Document) -> bool:
        """Update existing document"""
        try:
            if id in self.documents:
                document.updated_at = datetime.now()
                return self.delete_documents([id]) and self.add_documents([document])
            return False
        except Exception as e:
            raise DocumentStoreError(f"Error updating document: {str(e)}")