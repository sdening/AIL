# AIL: Agents in the Loop - Concept for automatically Prompting Large Language Models 
* Authors: Group of 5 in NLP Lab Course
* Prompt Optimization NLP Lab Course
* **Note:** This is preliminary version of our code. The complete code to run all experiments in the paper will be added shortly.

<img src="./Main Pipeline.png" alt="teaser image" width="7500"/>

## Dependencies
...

## Installation
The simplest way to run our code is to start with a fresh environment.
```
conda create -n AIL python=3.9
source activate AIL
pip install -r requirements.txt
```

# Tools
## Add / Use examples tool 
#### Description: 
Tool allows to:
  - add positive examples to the prompt
  - add negative examples to the prompt
  - remove positive examples from the prompt
  - remove negative examples from the prompt

#### Function: 
```
Inputs:
    current_clause (str): The current_clause (reference)
    df_test (df): The test set containing other clauses
    cat (str): The category name
    example_type (str): The type of example to find ('positive' or 'negative').
    prompt_ (str): The prompt to which the example will be added or from which it will be removed.
    mode (str): Mode of operation - 'add' to add an example, 'remove' to remove an example.

Returns:
    str: The updated prompt which contains the current examples of the global dictionary.

Function Call: 
    find_most_similar_bm25(current_clause: str, df_test: df, cat: str, example_type: str, prompt_: str, mode: str)
  
Variables that are used:
global added_examples: added_examples = {
                          "positive": [],
                          "negative": []
                        }

-> This example dictionary will be maintained thoughout the loops to make sure prompt is always up to date
```
#### Example usage in function: "test_prompt" or "get_prediction"
```
...
    for i, ex in data.iterrows():
        
        legal_std = legal_standards[cat]['fairness_q'] 
    
        prompt_ = 'Consider the following online terms of service clause: "' + ex['text'] + ' \n ' + legal_std +' '+ prompt

        modified_prompt = find_most_similar_bm25(ex['text'], df_test, cat, "positive", prompt_, "add")
...



```

## Reformat tool 
#### Description: 
Tool allows to reformat text into list of shot sentences, each one begins with "-".
The content can be changed to save grammarly correct in each bulletpoint.
#### Function: 
```
Inputs:
    prompt_to_modify(str): String text of prompt

Returns:
    str: The updated prompt text in new format of shot sentences constructed based on the given content

Function Call: 
    reformat(input_text: str)
  
Overview how is the fuction built:

def reformat(input_text: str) -> str:
    # runner = ReformatRunner()
    device = 0 if torch.cuda.is_available() else -1
    # pipe = pipeline("text2text-generation", model="Isotonic/bullet-points-generator", device=device) 
    model_name = "Isotonic/bullet-points-generator"
    pipe = pipeline(
        "text2text-generation", 
        model=model_name, 
        device=device,
        num_beams=5,                # Use beam search with 5 beams
        temperature=1.0,            # Set temperature to 1.0
        top_k=50,                   # Use top-k sampling
        repetition_penalty=1.2,     # Set repetition penalty
    )
    reformatted_text = pipe(input_text, max_length=100, clean_up_tokenization_spaces=False)[0]['generated_text']
    # print(repr(reformatted_text))
    sentences = re.split(r'n-', reformatted_text)
    rewritten = "\n".join([f" - {sentence.strip()}" for sentence in sentences if sentence.strip()])

    return rewritten

```

#### Example usage in function:
```
...
    starting_prompt = "Start with \"yes\" or \"no\" and then with \"yes\" or \"no\" with \"yes\" or \"no\" justify your response in no more than 50 words."
    reformatted_text = reformat(starting_prompt)
    print(repr(reformatted_text))
    ' - Start with "yes" or "no"\n - Repeat with "yes" or "no"\n - Explanation: "yes" or "no"\n - Explanation: "yes" or "no" in no more than 50 words'

    starting_prompt =  "Start with \"yes\" or \"no\" and then justify your response in no more than 50 words."
    reformatted_text = reformat(starting_prompt)
    print(repr(reformatted_text))
    ' - Start with "yes" or "no"\n - Justify response in no more than 50 words'

...
```

# Scorers

## score_complete
#### Description:
    Gives weightes score back. Score is between 0 and 100. 
    Weights score of performance, length and readability. 
#### Function
```
    Parameters:
    - intro(str): Introduction of prompt 
    - legal_description (str): Description of categorie
    - pos_examples (list[str]): List of positive examples.
    - neg_examples (list[str]): List of negative examples.  
    - modified_prompt (str): Modified prompt, output of edit tools.
    - initial_prompt (str): Inital prompt, first prompt evere used.
    - df (pd.DataFrame): DataFrame containing all Datapoints that should be scored on.
    - cat (str): Categorie that is to be tested
    - groq_client: Groq client that is used in main. 
    - weight (dict[float]) = {'p':0.8, 'r':0.1, 'l':0.1}: Give ratio for weighting. p = performance, r=readability, l=lenght 

    Returns:
    - score (int): The compelte and weighted score.
```
#### Example usage in function:
```
...
    score = score(intro, legal_description, pos_examples, neg_examples, modified_prompt, initial_prompt, df, cat, groq_client, weight)
...
```


## score_length
#### Description:
    Score the prompt based on its length in relation to the initial prompt length.
#### Function
```
    Parameters:
    - modified_prompt_text (str): The modified prompt text to be scored.
    - initial_prompt_length (int): The assumed length of the initial prompt text. Default is 100.

    Returns:
    - score (int): The score based on the length comparison.
```
#### Example usage in function:
```
...
    modified_prompt_text = "Start  with 'yes' or 'no' and then justify your response in no more than 50 words."
    initial_prompt_length = len("Start your answer with 'yes' or 'no' and then justify your response in no more than 50 words.")

    score = score_prompt_length(modified_prompt_text, initial_prompt_length)
...
```
