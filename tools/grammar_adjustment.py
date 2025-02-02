from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

class GrammarAdjustment():
    def __init__(self) -> None:
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
        self.model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

    def grammatic_adjustment(self, inital_prompt: str):
        gram_check_promtp = f"Fix grammatical errors in this sentence: {inital_prompt}"
        input_ids = self.tokenizer(gram_check_promtp, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=256)
        edited_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return edited_text



        

