import spacy
from typing import List, Dict

class EntityRecognizer:
    def __init__(self):
        # Load small English model
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, query: str) -> List[Dict]:
        """Extract named entities from query"""
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return entities

    def get_entity_types(self, query: str) -> Dict[str, List[str]]:
        """Group entities by their types"""
        doc = self.nlp(query)
        entity_types = {}
        
        for ent in doc.ents:
            if ent.label_ not in entity_types:
                entity_types[ent.label_] = []
            entity_types[ent.label_].append(ent.text)
            
        return entity_types