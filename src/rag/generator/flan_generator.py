from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict
from .base_generator import BaseGenerator
from ..document_store.document import Document

class FlanGenerator(BaseGenerator):
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_context_length = 512
        self.max_length = 256

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.content}")
        return "\n\n".join(context_parts)

    def generate(self, query: str, contexts: List[Document], enhanced_query: str = None) -> Dict:
        """Generate answer using retrieved contexts"""
        # Format prompt
        context_text = self._format_context(contexts)
        prompt = f"""Answer the query based on the following context.
        
        Context:
        {context_text}
        
        Query: {query}
        Enhanced Query: {enhanced_query or query}
        
        Answer:"""

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_context_length, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=3
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'answer': answer,
            'sources': [doc.id for doc in contexts],
            'scores': [doc.score for doc in contexts]
        }