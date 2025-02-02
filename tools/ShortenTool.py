import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

class ShortenTool:
    def __init__(self):
        self.stopwords = set(stopwords.words('english')) #init of stopwords in english

    def shorten_prompt(self, prompt):
        """
        Shortens given prompt using stopwords remover via nltk. 

        Parameters:
        prompt (str): The prompt to shorten.

        Returns:
        str: The shortened prompt.
        """
        prompt = self.remove_stopwords(prompt)
        return prompt

    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words) #concatenate the words which are not stopwords

#ex usage
if __name__ == "__main__":
    prompt = 'Start your answer with "yes" or "no" and then justify your response in no more than 50 words.'
    shortener = PromptShortener()
    shortened_prompt = shortener.shorten_prompt(prompt)
    print("Original prompt len:", len(prompt))
    print("Shortened prompt:", shortened_prompt)
    print("Shortened prompt len:", len(shortened_prompt))