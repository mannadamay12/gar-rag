from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime

@dataclass
class Document:
    """Class representing a document or text chunk"""
    
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None
    
    # Tracking fields
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate document after initialization"""
        if not self.id or not self.content:
            raise ValueError("Document must have both id and content")
        
        if self.embedding is not None:
            if not isinstance(self.embedding, np.ndarray):
                raise ValueError("Embedding must be a numpy array")

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'score': self.score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, doc_dict: Dict[str, Any]) -> 'Document':
        """Create document from dictionary"""
        if doc_dict.get('embedding'):
            doc_dict['embedding'] = np.array(doc_dict['embedding'])
        return cls(**doc_dict)