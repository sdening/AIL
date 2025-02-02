from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer
import torch

def paraphrase_huggingface(inital_prompt: str, pipe: pipeline) -> str:
    """
    Description = "Generates a variation of the input text while keeping the semantic meaning using HuggingFace pipeline."
    """
    name = "paraphrase_huggingface"

    hf = HuggingFacePipeline(pipeline=pipe)

    # Prepare the prompt template for the paraphrasing task
    template = """Generate a variation of the input text while keeping the semantic meaning: \n Input:{text} \n Output: """

    # Create the chain and run it
    prompt = PromptTemplate.from_template(template)
    chain = prompt | hf.bind(skip_prompt=True)
    print(f"Starting to paraphrase the following prompt: {inital_prompt}\n Using {pipeline} as pipeline:")
    result = chain.invoke({"text": inital_prompt})
    
    return result

class ParaphrasePegasus():
    def __init__(self, model_name: str = 'tuner007/pegasus_paraphrase') -> None:    
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.para_tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.para_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)

    def paraphrase_pegasus(self, inital_prompt: str, num_beams: int = 10, num_return_sequences: int = 1) -> str:
        """
        Description: "Generates a variation of the input text while keeping the semantic meaning using Pegasus model."
        """
        # Paraphrase pipeline
        print(f"Starting to paraphrase the following prompt: {inital_prompt} \n Using Pegasus:")
        batch = self.para_tokenizer([inital_prompt], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.para_model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5, do_sample=True)
        tgt_text = self.para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return tgt_text[0]  