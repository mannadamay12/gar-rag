from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseLanguageModel
from typing import List

class FalconModel(BaseLanguageModel):
    def __init__(self, model_name="tiiuae/falcon-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        return [self.tokenizer.decode(out, skip_special_tokens=True) 
                for out in outputs]