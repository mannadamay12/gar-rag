from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class BaseEmbedder(ABC):
    """Base class for text embedding models"""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for given text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        pass

    @property
    def embedding_dim(self) -> int:
        """Return dimension of embeddings"""
        raise NotImplementedError