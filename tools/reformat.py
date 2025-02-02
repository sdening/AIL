import re
import torch
import transformers
from transformers import pipeline
import logging

# Configure logging to suppress specific warnings during model loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
class Reformat():
    def __init__(self) -> None:
        self.device = 0 if torch.cuda.is_available() else -1
        # pipe = pipeline("text2text-generation", model="Isotonic/bullet-points-generator", device=device) 
        self.model_name = "Isotonic/bullet-points-generator"
        self.pipe = pipeline(
            "text2text-generation", 
            model=self.model_name, 
            device=self.device,
            num_beams=5,                # Use beam search with 5 beams
            temperature=1.0,            # Set temperature to 1.0
            top_k=50,                   # Use top-k sampling
            repetition_penalty=1.2,     # Set repetition penalty
        )
    def reformat(self, input_text: str) -> str:
        reformatted_text = self.pipe(input_text, max_length=100, clean_up_tokenization_spaces=True)[0]['generated_text']
        # print(repr(reformatted_text))
        sentences = re.split(r'n-', reformatted_text)
        rewritten = "\n".join([f" - {sentence.strip()}" for sentence in sentences if sentence.strip()])

        return rewritten

# Define example usage function
def example_usage():
    # Initial prompt text
    starting_prompt =  "Start with \"yes\" or \"no\" and then justify your response in no more than 50 words."
    reformat = Reformat()
    reformatted_text = reformat.reformat(starting_prompt)
    
    # print(f"{reformatted_text}")
    print(repr(reformatted_text))

# Run example usage
example_usage()
