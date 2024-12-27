from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import re
import nltk
import logging
from ..document_store.document import Document

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 512              # Target chunk size in tokens
    chunk_overlap: int = 50            # Overlap between chunks
    length_function: str = "token"     # "token" or "character"
    min_chunk_size: int = 50           # Minimum chunk size
    max_chunk_size: int = 1024         # Maximum chunk size
    respect_sentence_boundaries: bool = True  # Try to break at sentence boundaries
    skip_empty_lines: bool = True      # Skip empty lines when chunking
    clean_whitespace: bool = True      # Clean excessive whitespace
    remove_urls: bool = True           # Remove URLs from text
    keep_table_structure: bool = True  # Try to maintain table structures

class DocumentChunker:
    """Handles document preprocessing and chunking"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize document chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")

    def process_document(self, 
                        content: Union[str, Dict],
                        doc_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> List[Document]:
        """
        Process and chunk document content
        
        Args:
            content: Document content or dict with text fields
            doc_id: Document ID
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
            
            # Split into chunks
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
                    "chunk_size": len(chunk.split()),
                    "source_id": doc_id
                })
                
                doc = Document(
                    id=chunk_id,
                    content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _extract_text_from_dict(self, content: Dict) -> str:
        """Extract text from dictionary fields"""
        text_fields = []
        
        # Priority fields to check
        priority_fields = ['text', 'content', 'body', 'description', 'title']
        
        # First check priority fields
        for field in priority_fields:
            if field in content and content[field]:
                text_fields.append(str(content[field]))
        
        # Then check other string fields
        for key, value in content.items():
            if (isinstance(value, str) and value and 
                key not in priority_fields):
                text_fields.append(value)
        
        return "\n\n".join(text_fields)

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
            
        # Remove URLs if configured
        if self.config.remove_urls:
            text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Clean whitespace if configured
        if self.config.clean_whitespace:
            # Remove excessive newlines while preserving paragraph breaks
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Normalize other whitespace
            text = re.sub(r'\s+', ' ', text)
            
        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []
            
        chunks = []
        
        # First split into sentences if configured
        if self.config.respect_sentence_boundaries:
            sentences = nltk.sent_tokenize(text)
        else:
            sentences = [text]
            
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty sentences if configured
            if not sentence and self.config.skip_empty_lines:
                continue
                
            # Calculate length based on configuration
            if self.config.length_function == "token":
                sentence_length = len(sentence.split())
            else:
                sentence_length = len(sentence)
            
            # Handle sentences longer than max chunk size
            if sentence_length > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Split long sentence
                sentence_chunks = self._split_long_sentence(sentence)
                chunks.extend(sentence_chunks)
                continue
            
            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length > self.config.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Add overlap if needed
        if self.config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split long sentence into smaller chunks"""
        if self.config.length_function == "token":
            words = sentence.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + 1 > self.config.max_chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = 1
                else:
                    current_chunk.append(word)
                    current_length += 1
                    
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
            return chunks
        else:
            # Character-based splitting
            return [sentence[i:i + self.config.max_chunk_size] 
                   for i in range(0, len(sentence), self.config.max_chunk_size)]

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
            if self.config.length_function == "token":
                prev_words = chunks[i-1].split()
                current_words = chunks[i].split()
                overlap_tokens = prev_words[-self.config.chunk_overlap:]
                merged_chunk = overlap_tokens + current_words
                overlapped_chunks.append(" ".join(merged_chunk))
            else:
                # Character-based overlap
                overlap_text = chunks[i-1][-self.config.chunk_overlap:]
                overlapped_chunks.append(overlap_text + chunks[i])
                
        return overlapped_chunks

    def estimate_chunks(self, text: str) -> Tuple[int, int]:
        """
        Estimate number of chunks and tokens for a text
        
        Returns:
            Tuple of (estimated_chunks, total_tokens)
        """
        if self.config.length_function == "token":
            total_tokens = len(text.split())
        else:
            total_tokens = len(text)
            
        estimated_chunks = (total_tokens // 
                          (self.config.chunk_size - self.config.chunk_overlap)) + 1
                          
        return estimated_chunks, total_tokens