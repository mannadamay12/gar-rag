import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from src.rag.document_store.faiss_store import FAISSDocumentStore
from src.rag.document_store.base_store import Document

@pytest.fixture(scope="session")
def encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@pytest.fixture(scope="session")
def store():
    return FAISSDocumentStore(embedding_dim=384)  # MiniLM embedding dimension

@pytest.fixture(scope="session")
def msmarco_samples():
    # Load small subset of MSMARCO
    dataset = load_dataset("ms_marco", "v2.1", split="train[:1000]")
    return dataset

@pytest.fixture(scope="session")
def beir_samples():
    # Load NFCorpus from BEIR
    dataset = load_dataset("BeIR/nfcorpus", split="test[:1000]")
    return dataset