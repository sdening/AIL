import tools.score_length  as sl
import tools.score_readability as sr
import tools.score_performance as sp
import pandas as pd

def score_prompt(intro: str, pos_examples: list[str], neg_examples: list[str], modified_prompt: str, initial_prompt: str, df: pd.DataFrame, cat:str, groq_client, weight: dict[float] = {'p':1.0, 'r':0, 'l':0.0}) -> int:

    #readability_rater = sr.TextUnderstandabilityRater()
    scorer= sp.PerformanceScorer(df, cat, intro, modified_prompt, groq_client, neg_examples, pos_examples, verbose=True)
    print("Start getting performance score")
    performance_score, token_len = scorer.f1()
    print(f"\nPerformance score is: {performance_score}")
    print("Start getting length score")
    length_score = sl.score_prompt_length(modified_prompt, initial_prompt)
    print(f"Lenght score is: {length_score}")
    #print(f"Start getting readability score")
    #readability_score = readability_rater.rate_text(modified_prompt)
    #print(f"Readability score is: {readability_score}")
    return weight['p']*performance_score + weight['l']*length_score, token_len
 
def score_prompt_01(intro: str, pos_examples: list[str], neg_examples: list[str], modified_prompt: str, initial_prompt: str, df: pd.DataFrame, cat:str, groq_client, weight: dict[float] = {'p':1.0, 'r':0, 'l':0.0}) -> int:

    #readability_rater = sr.TextUnderstandabilityRater()
    scorer= sp.PerformanceScorer(df, cat, intro, modified_prompt, groq_client, neg_examples, pos_examples, verbose=True)
    print("Start getting performance score")
    performance_score, token_len = scorer.f1()
    print(f"\nPerformance score is: {performance_score}")
    print("Start getting length score")
    length_score = sl.score_prompt_length(modified_prompt, initial_prompt)
    print(f"Lenght score is: {length_score}")
    #print(f"Start getting readability score")
    #readability_score = readability_rater.rate_text(modified_prompt)
    #print(f"Readability score is: {readability_score}")
    return (weight['p']*performance_score + weight['l']*length_score)/100, token_len

def score_prompt_11(intro: str, pos_examples: list[str], neg_examples: list[str], modified_prompt: str, initial_prompt: str, df: pd.DataFrame, cat:str, groq_client, weight: dict[float] = {'p':1.0, 'r':0, 'l':0.0}) -> int:

    #readability_rater = sr.TextUnderstandabilityRater()
    scorer= sp.PerformanceScorer(df, cat, intro, modified_prompt, groq_client, neg_examples, pos_examples, verbose=True)
    print("Start getting performance score")
    performance_score, token_len = scorer.f1()
    print(f"\nPerformance score is: {performance_score}")
    print("Start getting length score")
    length_score = sl.score_prompt_length(modified_prompt, initial_prompt)
    print(f"Lenght score is: {length_score}")
    #print(f"Start getting readability score")
    #readability_score = readability_rater.rate_text(modified_prompt)
    #print(f"Readability score is: {readability_score}")
    return ((weight['p']*performance_score + weight['l']*length_score)/50)-1, token_len
 