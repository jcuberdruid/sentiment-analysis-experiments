import numpy as np
import spacy
from gensim.models import Word2Vec
from multiprocessing import Pool
from data_handler import Data_handler

def tokenizer_init():
    global nlp
    nlp = spacy.load('en_core_web_sm')

def tokenize(string):
    doc = nlp(string.lower())
    return [token.text for token in doc if token.is_alpha]

class Preprocessor:
    def __init__(self, data):
        self.processed_data = []
        self.preprocess(data)

    def preprocess(self, data):
        with Pool(14, initializer=tokenizer_init) as p:
            tokenized_data = p.map(tokenize, data)
        self.data_w2v = Word2Vec(tokenized_data, vector_size=300, window=10, min_count=1, sample=1e-3, workers=14)
        vectorized_data = []
        for doc in tokenized_data:
            doc_vector = [self.data_w2v.wv[word] if word in self.data_w2v.wv else np.zeros(300) for word in doc]
            vectorized_data.append(doc_vector)
        max_len = max(len(doc) for doc in vectorized_data)
        padded_data = np.array([np.pad(doc, ((0, max_len - len(doc)), (0, 0)), mode='constant') for doc in vectorized_data])
        np.save('word_vectors.npy', padded_data)

load = Data_handler("../data/IMDB_Dataset.csv")
proc = Preprocessor(load.data)


