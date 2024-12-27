# tests/rag/conftest.py
import pytest
import numpy as np
from src.rag.document_store.document import Document
from src.rag.pipeline import RAGPipeline
from src.rag.embeddings.sentence_embedder import SentenceEmbedder
from src.rag.generator.flan_generator import FlanGenerator

@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            id="1",
            content="Climate change is causing global temperatures to rise.",
            embedding=np.random.rand(768),
            metadata={"source": "science_journal"}
        ),
        Document(
            id="2",
            content="Renewable energy sources can help reduce carbon emissions.",
            embedding=np.random.rand(768),
            metadata={"source": "energy_report"}
        ),
        Document(
            id="3",
            content="Electric vehicles are becoming more popular in urban areas.",
            embedding=np.random.rand(768),
            metadata={"source": "transport_study"}
        )
    ]

@pytest.fixture
def mock_embedder(mocker):
    """Mock embedder for testing"""
    mock = mocker.Mock(spec=SentenceEmbedder)
    mock.embedding_dim = 768
    mock.embed.return_value = np.random.rand(768)
    mock.embed_batch.return_value = np.random.rand(3, 768)
    return mock

@pytest.fixture
def mock_generator(mocker):
    """Mock generator for testing"""
    mock = mocker.Mock(spec=FlanGenerator)
    mock.generate.return_value = {
        'answer': "Generated answer about climate change",
        'sources': ["1", "2"],
        'scores': [0.9, 0.8]
    }
    return mock

@pytest.fixture
def rag_pipeline(mock_embedder, mock_generator):
    """Create RAG pipeline with mock components"""
    pipeline = RAGPipeline()
    pipeline.embedder = mock_embedder
    pipeline.generator = mock_generator
    return pipeline