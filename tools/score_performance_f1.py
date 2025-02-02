import pandas as pd
import re
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import entropy
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from nltk.tokenize import word_tokenize
import time

class PerformanceScorer():
    def __init__(self, data: pd.DataFrame, cat: str, intro:str, modified_prompt: str, groq_client, legal_clause, neg_examples: list[str] = [], pos_examples: list[str]=[], verbose: bool = False):
        self.data = data
        self.cat = cat
        self.intro = intro
        self.modified_prompt = modified_prompt
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.verbose = verbose
        self.client = groq_client
        self.sys_message_activ = False
        self.legal_standards = legal_clause
        
    def get_prediction(self) -> int:
        yes_res = [r'^[\s"]?[Yy]es[\.,\s]']
        pred = []
        actual_label = []
        mean_token_len = 0
        token_len = 0
        
        prompt = ' ' + self.legal_standards

        if len(self.neg_examples) > 0: 
            prompt = prompt + ' For example, consider this clause which is not of this category: "' + '"; "'.join(example for example in self.neg_examples) + '"'

        if len(self.pos_examples) > 0:
            prompt = prompt + ' For example, consider this clause of the same category: "' + '"; "'.join(example for example in self.pos_examples) + '"'
        
        prompt = prompt + ' ' + self.modified_prompt
        
        if self.verbose:
            print("Start analyzing predicitions:")
        i = 0
        for _, ex in self.data.iterrows(): 
              
            prompt = self.intro + ' ' + ex['text'] + prompt
            
            if self.verbose:
                print(".", end='')
            token_len = token_len + len(word_tokenize(prompt))
            message = HumanMessage(content=prompt)
            if i%20 == 0:
                time.sleep(3) 
            if self.sys_message_activ:
                sys_message = SystemMessage(content="Start your answer with \'yes\' or \'no\'.")
                gen_text = self.client.invoke([sys_message, message]).content   
            else:
                gen_text = self.client.invoke([message]).content 
            pred.append(1) if re.search(yes_res[0], gen_text) is not None else pred.append(0) 
            actual_label.append(ex[self.cat])
            i = i + 1

        mean_token_len = token_len/self.data.shape[0]   
        return pred, actual_label, mean_token_len
    
    def score(self):
        y_pred, y_true, _ = self.get_prediction()
        raw_acc = balanced_accuracy_score(y_true, y_pred)
        unique_labels = set(y_true)  
        label_frequencies = [y_pred.count(l) / len(y_pred) for l in unique_labels]
        
        score_ = np.round(100*raw_acc, 2) + 10*entropy(label_frequencies)
        
        return score_
    
    def f1(self):
        y_pred, y_true, mean_token_len = self.get_prediction()
        return f1_score(y_true, y_pred, average='macro')*100, mean_token_len