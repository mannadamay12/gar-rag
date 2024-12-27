# tests/conftest.py
import pytest
from sentence_transformers import SentenceTransformer
from src.rag.document_store.faiss_store import FAISSDocumentStore

@pytest.fixture(scope="session")
def encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@pytest.fixture(scope="session")
def store():
    return FAISSDocumentStore(embedding_dim=384)  # MiniLM embedding dimension