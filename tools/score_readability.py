import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class TextUnderstandabilityRater:
    def __init__(self):
        # Initialize the HuggingFace models for grammar correctio        
        self.tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-xl")
        self.model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-xl")

    def rate_text(self, text):
        # Step 1: Preprocessing
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)

        # Step 2: Grammar Check (using HuggingFace model and cosine similarity)
        grammar_score = self.check_grammar(text)

        # Step 4: Semantic Coherence
        coherence_score = self.check_coherence(tokens)

        # Combine scores for final rating (weights can be adjusted)
        final_score = grammar_score #+ 0.3 * coherence_score
        return final_score

    def check_grammar(self, text):
        # Generate corrected text using the HuggingFace model
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=256)
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate Levenshtein Distance between original and corrected text
        distance = self.levenshtein_distance(text, corrected_text)
        max_length = max(len(text), len(corrected_text))
        similarity_score = (1 - (distance / max_length)) * 100
        

        print(f"similarity_score is {similarity_score} \n for input: {text} \n and output: {corrected_text}")
        return similarity_score

    def levenshtein_distance(self, s, t):
        m, n = len(s), len(t)
        if m < n:
            s, t = t, s
            m, n = n, m
        d = [list(range(n + 1))] + [[i] + [0] * n for i in range(1, m + 1)]
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if s[i - 1] == t[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
        return d[m][n]

    def check_coherence(self, tokens):
        # Simplified semantic coherence check
        tagged = nltk.pos_tag(tokens)
        nouns_verbs = [word for word, pos in tagged if pos in ['NN', 'VB', 'VBP', 'VBZ']]
        
        # Assume high coherence if more than 50% of words are meaningful (nouns, verbs)
        coherence_score = len(nouns_verbs) / len(tokens) * 100
        return coherence_score

