from abc import ABC, abstractmethod
from typing import List, Dict
from ..document_store.document import Document

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, contexts: List[Document], enhanced_query: str = None) -> Dict:
        pass