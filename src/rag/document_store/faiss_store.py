import faiss
import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict
import logging
from .base_store import BaseDocumentStore, Document, DocumentStoreError

logger = logging.getLogger(__name__)

class FAISSDocumentStore(BaseDocumentStore):
    """FAISS-based document store implementation"""
    
    def __init__(self,
                 embedding_dim: int = 768,
                 index_name: str = "document_store",
                 similarity_metric: str = "cosine",
                 **kwargs):
        """Initialize FAISS document store
        
        Args:
            embedding_dim: Dimension of document embeddings
            index_name: Name of the index
            similarity_metric: 'cosine' or 'l2' distance
            **kwargs: Additional parameters
        """
        self.similarity_metric = similarity_metric
        super().__init__(embedding_dim=embedding_dim, index_name=index_name, **kwargs)
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        # Mapping of doc IDs to FAISS internal IDs
        self.docid_to_faissid: Dict[str, int] = {}
        self.faissid_to_docid: Dict[int, str] = {}
        self.next_faiss_id = 0

    def _initialize_store(self, **kwargs) -> None:
        """Initialize FAISS index"""
        if self.similarity_metric == "cosine":
            # Normalize vectors + use inner product
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_documents(self, 
                     documents: List[Document],
                     batch_size: int = 1000) -> bool:
        """Add documents to store
        
        Args:
            documents: List of documents to add
            batch_size: Size of batches for adding documents
            
        Returns:
            bool: True if successful
        """
        try:
            self._validate_documents(documents)
            
            # Process in batches
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
                        # Map document ID to FAISS ID
                        self.docid_to_faissid[doc.id] = self.next_faiss_id
                        self.faissid_to_docid[self.next_faiss_id] = doc.id
                        
                        # Prepare embedding for FAISS
                        if self.similarity_metric == "cosine":
                            embedding = doc.embedding / np.linalg.norm(doc.embedding)
                        else:
                            embedding = doc.embedding
                        embeddings_batch.append(embedding)
                        
                        self.next_faiss_id += 1
                
                if embeddings_batch:
                    embeddings_array = np.array(embeddings_batch)
                    self.index.add(embeddings_array)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise DocumentStoreError(f"Failed to add documents: {str(e)}")

    def get_document_by_id(self, id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        return self.documents.get(id)

    def search_documents(self,
                        query_embedding: np.ndarray,
                        top_k: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of documents to retrieve
            filters: Metadata filters
            
        Returns:
            List of Documents sorted by similarity
        """
        try:
            self.validate_embedding(query_embedding)
            
            # Normalize query vector for cosine similarity
            if self.similarity_metric == "cosine":
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
            # Reshape query to 2D array
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS index
            scores, faiss_ids = self.index.search(query_embedding, top_k)
            
            # Convert results to documents
            results = []
            for score, faiss_id in zip(scores[0], faiss_ids[0]):
                if faiss_id == -1:  # FAISS returns -1 if not enough results
                    continue
                    
                doc_id = self.faissid_to_docid[faiss_id]
                doc = self.documents[doc_id]
                
                # Apply filters if specified
                if filters and not self._check_filters(doc, filters):
                    continue
                    
                doc.score = float(score)
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise DocumentStoreError(f"Search failed: {str(e)}")

    def update_embeddings(self, documents: List[Document]) -> bool:
        """Update embeddings for existing documents"""
        try:
            # Remove old embeddings
            doc_ids_to_update = [doc.id for doc in documents]
            faiss_ids_to_remove = [self.docid_to_faissid[doc_id] 
                                 for doc_id in doc_ids_to_update 
                                 if doc_id in self.docid_to_faissid]
            
            if faiss_ids_to_remove:
                # Currently, FAISS doesn't support removal of individual vectors
                # We need to rebuild the index
                self._rebuild_index_without_ids(faiss_ids_to_remove)
            
            # Add new embeddings
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {str(e)}")
            raise DocumentStoreError(f"Failed to update embeddings: {str(e)}")

    def get_documents_by_filters(self,
                               filters: Dict[str, Any],
                               limit: Optional[int] = None) -> List[Document]:
        """Get documents matching metadata filters"""
        matching_docs = []
        for doc in self.documents.values():
            if self._check_filters(doc, filters):
                matching_docs.append(doc)
                if limit and len(matching_docs) >= limit:
                    break
        return matching_docs

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from store"""
        try:
            faiss_ids_to_remove = []
            for doc_id in ids:
                if doc_id in self.docid_to_faissid:
                    faiss_ids_to_remove.append(self.docid_to_faissid[doc_id])
                    del self.documents[doc_id]
                    del self.docid_to_faissid[doc_id]
            
            if faiss_ids_to_remove:
                self._rebuild_index_without_ids(faiss_ids_to_remove)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise DocumentStoreError(f"Failed to delete documents: {str(e)}")

    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)

    def clear(self) -> bool:
        """Clear all documents from store"""
        try:
            self.documents.clear()
            self.docid_to_faissid.clear()
            self.faissid_to_docid.clear()
            self.next_faiss_id = 0
            self._initialize_store()
            return True
        except Exception as e:
            logger.error(f"Error clearing store: {str(e)}")
            raise DocumentStoreError(f"Failed to clear store: {str(e)}")

    def _check_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches filters"""
        for key, value in filters.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False
        return True

    def _rebuild_index_without_ids(self, faiss_ids_to_remove: List[int]) -> None:
        """Rebuild FAISS index excluding specified IDs"""
        # Get remaining documents
        remaining_embeddings = []
        new_faissid_to_docid = {}
        new_docid_to_faissid = {}
        new_faiss_id = 0
        
        for old_faiss_id, doc_id in self.faissid_to_docid.items():
            if old_faiss_id not in faiss_ids_to_remove:
                doc = self.documents[doc_id]
                embedding = doc.embedding
                if self.similarity_metric == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)
                remaining_embeddings.append(embedding)
                
                new_faissid_to_docid[new_faiss_id] = doc_id
                new_docid_to_faissid[doc_id] = new_faiss_id
                new_faiss_id += 1
        
        # Rebuild index
        self._initialize_store()
        if remaining_embeddings:
            self.index.add(np.array(remaining_embeddings))
        
        # Update mappings
        self.faissid_to_docid = new_faissid_to_docid
        self.docid_to_faissid = new_docid_to_faissid
        self.next_faiss_id = new_faiss_id