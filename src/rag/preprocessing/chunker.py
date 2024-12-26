from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import re
import nltk
from nltk.tokenize import sent_tokenize
import logging
from ..document_store.base_store import Document

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 512                  # Target chunk size in tokens
    chunk_overlap: int = 50                # Overlap between chunks
    length_function: str = "token_count"   # "token_count" or "char_count"
    separator: str = " "                   # Chunk separator
    min_chunk_size: int = 50              # Minimum chunk size
    skip_empty_lines: bool = True         # Skip empty lines when chunking

class DocumentPreprocessor:
    """Handles document preprocessing including chunking and cleaning"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")

    def preprocess_document(self, 
                          content: Union[str, Dict],
                          doc_id: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> List[Document]:
        """
        Preprocess document content into chunks
        
        Args:
            content: Document content or dict with text fields
            doc_id: Base document ID
            metadata: Additional metadata
            
        Returns:
            List of Document objects
        """
        try:
            # Extract text if content is a dict
            if isinstance(content, dict):
                text = self._extract_text_from_dict(content)
            else:
                text = content

            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Create chunks
            chunks = self._create_chunks(cleaned_text)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}" if doc_id else f"chunk_{i}"
                
                # Add chunk metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.split())
                })
                
                doc = Document(
                    id=chunk_id,
                    content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error preprocessing document: {str(e)}")
            raise

    def _extract_text_from_dict(self, content: Dict) -> str:
        """Extract text from dictionary fields"""
        text_fields = []
        
        # Priority fields to check
        priority_fields = ['text', 'content', 'body', 'description']
        
        # First check priority fields
        for field in priority_fields:
            if field in content:
                text_fields.append(str(content[field]))
        
        # Then check other string fields
        for key, value in content.items():
            if isinstance(value, str) and key not in priority_fields:
                text_fields.append(value)
        
        return "\n".join(text_fields)

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        
        # Split into sentences first
        sentences = sent_tokenize(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty sentences if configured
            if not sentence and self.config.skip_empty_lines:
                continue
            
            # Calculate length based on configuration
            if self.config.length_function == "token_count":
                sentence_length = len(sentence.split())
            else:
                sentence_length = len(sentence)
            
            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length > self.config.chunk_size:
                if current_chunk:
                    chunks.append(self.config.separator.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(self.config.separator.join(current_chunk))
        
        # Handle overlap if needed
        if self.config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks"""
        if len(chunks) <= 1:
            return chunks
            
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
                continue
                
            # Get overlap from previous chunk
            prev_chunk = chunks[i-1].split()
            current_chunk = chunks[i].split()
            
            overlap_tokens = prev_chunk[-self.config.chunk_overlap:]
            merged_chunk = overlap_tokens + current_chunk
            
            overlapped_chunks.append(' '.join(merged_chunk))
            
        return overlapped_chunks

    def estimate_chunks(self, text: str) -> Tuple[int, int]:
        """
        Estimate number of chunks and tokens
        
        Returns:
            Tuple of (estimated_chunks, total_tokens)
        """
        total_tokens = len(text.split())
        estimated_chunks = (total_tokens // self.config.chunk_size) + 1
        return estimated_chunks, total_tokens