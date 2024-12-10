import pytest
from src.query_processor.entity_recognizer import EntityRecognizer

@pytest.fixture
def recognizer():
    return EntityRecognizer()

def test_extract_entities(recognizer):
    query = "How does Tesla impact climate change in California?"
    entities = recognizer.extract_entities(query)
    
    # Should identify Tesla (ORG) and California (GPE)
    assert len(entities) >= 2
    assert any(e['text'] == 'Tesla' for e in entities)
    assert any(e['text'] == 'California' for e in entities)

def test_get_entity_types(recognizer):
    query = "What did Microsoft announce in New York last June?"
    entity_types = recognizer.get_entity_types(query)
    
    assert 'ORG' in entity_types
    assert 'GPE' in entity_types
    assert 'DATE' in entity_types