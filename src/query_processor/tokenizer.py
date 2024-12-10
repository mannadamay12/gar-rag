from typing import List, Dict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

class QueryTokenizer:
    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

    def _get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to WordNet POS tag"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def tokenize(self, text: str) -> Dict:
        """Advanced tokenization with POS tagging and lemmatization"""
        # Word tokens
        word_tokens = [word.lower() for word in word_tokenize(text)]
        
        # POS tagging
        pos_tags = nltk.pos_tag(word_tokens)
        
        # Lemmatization with POS tags
        lemmatized = [self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word)) 
                     for word in word_tokens]

        return {
            'word_tokens': word_tokens,
            'pos_tags': pos_tags,
            'lemmatized_tokens': lemmatized,
            'sentence_tokens': sent_tokenize(text)
        }

    def get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generate n-grams from tokens"""
        return [' '.join(tokens[i:i + n]) 
                for i in range(len(tokens) - n + 1)]