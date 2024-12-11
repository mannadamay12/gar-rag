# src/models/llama.py
from typing import List
from transformers import LlamaTokenizer, LlamaForCausalLM
from .base_model import BaseLanguageModel

class LlamaModel(BaseLanguageModel):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        return [self.tokenizer.decode(out, skip_special_tokens=True) 
                for out in outputs]