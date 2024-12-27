from typing import List, Union, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from .base_embedder import BaseEmbedder
import logging

logger = logging.getLogger(__name__)

class SentenceEmbedder(BaseEmbedder):
    """Sentence Transformer based embedder implementation"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cache_size: int = 100000):
        """
        Initialize sentence embedder
        
        Args:
            model_name: Name of sentence-transformer model
            device: Device to run model on ('cuda' or 'cpu')
            cache_size: Size of embedding cache
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model {model_name} on {device}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
        
        # Initialize cache
        self._cache = {}
        self._cache_size = cache_size

    def embed(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments for encoding
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
                
            # Check cache first
            cached_embeddings = []
            texts_to_embed = []
            indices = []
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached_embeddings.append(self._cache[text])
                else:
                    texts_to_embed.append(text)
                    indices.append(i)
            
            # Generate new embeddings if needed
            if texts_to_embed:
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    **kwargs
                )
                
                # Update cache
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    if len(self._cache) >= self._cache_size:
                        # Remove random item if cache full
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[text] = embedding
                
                # Combine cached and new embeddings
                embeddings = np.zeros((len(texts), self.embedding_dim))
                embeddings[indices] = new_embeddings
                if cached_embeddings:
                    non_indices = list(set(range(len(texts))) - set(indices))
                    embeddings[non_indices] = cached_embeddings
                    
            else:
                embeddings = np.array(cached_embeddings)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_batch(self, 
                   texts: List[str], 
                   batch_size: int = 32,
                   **kwargs) -> np.ndarray:
        """
        Generate embeddings in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches
            **kwargs: Additional arguments for encoding
            
        Returns:
            Numpy array of embeddings
        """
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embed(batch, **kwargs)
                all_embeddings.append(batch_embeddings)
                
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension"""
        return self._embedding_dim
        
    def save_cache(self, path: str):
        """Save embedding cache to file"""
        try:
            np.savez_compressed(
                path,
                texts=list(self._cache.keys()),
                embeddings=np.array(list(self._cache.values()))
            )
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            raise
            
    def load_cache(self, path: str):
        """Load embedding cache from file"""
        try:
            data = np.load(path, allow_pickle=True)
            self._cache = {
                text: emb for text, emb in zip(data['texts'], data['embeddings'])
            }
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()