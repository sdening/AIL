import pandas as pd
import re
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score 
from scipy.stats import entropy
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from nltk.tokenize import word_tokenize

class PerformanceScorer():
    def __init__(self, data: pd.DataFrame, cat: str, intro:str, modified_prompt: str, groq_client, neg_examples: list[str] = [], pos_examples: list[str]=[], verbose: bool = False):
        self.data = data
        self.cat = cat
        self.intro = intro
        self.modified_prompt = modified_prompt
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.verbose = verbose
        self.client = groq_client
        self.sys_message_activ = False
        self.legal_standards = {
                                                        'A': {
                                                            'fairness_q': 'Does this clause describe an arbitration dispute resolution process that is not fully optional to the consumer?'
                                                        },
                                                        'CH': {
                                                            'fairness_q': 'Does this clause specify conditions under which the service provider could amend and modify the terms of service and/or the service itself?'
                                                        },
                                                        'CR': {
                                                            'fairness_q': "Does this clause indicate conditions for content removal in the service provider's full discretion, and/or at any time for any or no reasons and/or without notice nor possibility to retrieve the content."
                                                        },
                                                        'J': {
                                                            'fairness_q': "Does this clause state that any judicial proceeding is to be conducted in a place other than the consumer's residence (i.e. in a different city, different country)?"
                                                        },
                                                        'LAW': {
                                                            'fairness_q': 'Does the clause define the applicable law as different from the law of the consumer’s country of residence?'
                                                        },
                                                        'LTD': {
                                                            'fairness_q': 'Does this clause stipulate that duties to pay damages by the provider are limited or excluded?'
                                                        },
                                                        'TER': {
                                                            'fairness_q': 'Does this clause stipulate that the service provider may suspend or terminate the service at any time for any or no reasons and/or without notice?'
                                                        },
                                                        'USE': {
                                                            'fairness_q': 'Does this clause stipulate that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them?'
                                                        },
                                                    }
  
    def get_prediction(self) -> int:
        yes_res = [r'^[\s"]?[Yy]es[\.,\s]']
        pred = []
        actual_label = []
        mean_token_len = 0
        token_len = 0
        
        prompt = ' ' + self.legal_standards[self.cat]['fairness_q']

        if len(self.neg_examples) > 0: 
            prompt = prompt + ' For example, consider this clause which is not of this category: "' + '"; "'.join(example for example in self.neg_examples) + '"'

        if len(self.pos_examples) > 0:
            prompt = prompt + ' For example, consider this clause of the same category: "' + '"; "'.join(example for example in self.pos_examples) + '"'
        
        prompt = prompt + ' ' + self.modified_prompt
        
        if self.verbose:
            print("Start analyzing predicitions:")
        for _, ex in self.data.iterrows(): 
              
            prompt = self.intro + ' ' + ex['text'] + prompt
            
            if self.verbose:
                print(".", end='')
            token_len = token_len + len(word_tokenize(prompt))
            message = HumanMessage(content=prompt)
            if self.sys_message_activ:
                sys_message = SystemMessage(content="Start your answer with \'yes\' or \'no\'.")
                gen_text = self.client.invoke([sys_message, message]).content   
            else:
                gen_text = self.client.invoke([message]).content 
            pred.append(1) if re.search(yes_res[0], gen_text) is not None else pred.append(0) 
            actual_label.append(ex[self.cat]) 
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
        return f1_score(y_true, y_pred)*100, mean_token_len
        #return recall_score(y_true, y_pred)*100, mean_token_len