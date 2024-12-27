# src/rag/pipeline.py
from typing import Dict, List, Optional
from .document_store.faiss_store import FAISSDocumentStore
from .embeddings.sentence_embedder import SentenceEmbedder
from .generator.flan_generator import FlanGenerator
from .ranking.ranker import ResultRanker
from .document_store.document import Document

class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        generator_model: str = "google/flan-t5-large",
        use_reranking: bool = True,
        top_k: int = 5
    ):
        self.embedder = SentenceEmbedder(model_name=embedding_model)
        self.document_store = FAISSDocumentStore(embedding_dim=self.embedder.embedding_dim)
        self.generator = FlanGenerator(model_name=generator_model)
        self.ranker = ResultRanker()
        self.use_reranking = use_reranking
        self.top_k = top_k

    def _retrieve(self, query: str, enhanced_query: Optional[str] = None) -> List[Document]:
        """Retrieve relevant documents"""
        # Use enhanced query for retrieval if available
        retrieval_query = enhanced_query or query
        
        # Get query embedding
        query_embedding = self.embedder.embed(retrieval_query)
        
        # Retrieve documents
        documents = self.document_store.search_documents(
            query_embedding,
            top_k=self.top_k
        )
        
        # Rerank if enabled
        if self.use_reranking and documents:
            documents = self.ranker.rerank(query, documents)
            
        return documents

    def process(
        self,
        query: str,
        enhanced_query: Optional[str] = None
    ) -> Dict:
        """Process query and generate answer"""
        # Retrieve relevant documents
        retrieved_docs = self._retrieve(query, enhanced_query)
        
        # Generate answer
        if not retrieved_docs:
            return {
                'answer': "No relevant information found.",
                'sources': [],
                'metadata': {
                    'original_query': query,
                    'enhanced_query': enhanced_query,
                    'num_docs': 0
                }
            }
            
        result = self.generator.generate(
            query=query,
            contexts=retrieved_docs,
            enhanced_query=enhanced_query
        )
        
        return {
            'answer': result['answer'],
            'sources': retrieved_docs,
            'metadata': {
                'original_query': query,
                'enhanced_query': enhanced_query,
                'num_docs': len(retrieved_docs),
                'scores': [doc.score for doc in retrieved_docs]
            }
        }