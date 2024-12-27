from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from .base_model import BaseLanguageModel

class FlanT5Model(BaseLanguageModel):
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)