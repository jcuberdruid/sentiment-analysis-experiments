import spacy
import numpy as np
import pandas as pd
import gensim.downloader as api
from multiprocessing import Pool
from dataset import DatasetLoader

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.preprocess()
    '''
    def preprocess(self):
        print("preprocessing and tokenizing...")
        nlp = spacy.load('en_core_web_sm', enable=["tagger", "attribute_ruler", "lemmatizer"])
        print("tokenizing with spacy")
        # Tokenize the data
        self.df['data'] = list(nlp.pipe(self.df['data'], n_process=8, batch_size=5000))
        self.df['data'] = self.df['data'].map(lambda doc: [token.text for token in doc if token.is_alpha and not token.is_stop])
        #remove largest
        self.df['token_count'] = self.df['data'].map(len)
        indexes_to_remove = self.df.nlargest(10, 'token_count').index
        self.df = self.df.drop(indexes_to_remove)
        self.process_vectors()
    def preprocess(self): #same but removes proper names 
        print("preprocessing and tokenizing...")
        nlp = spacy.load('en_core_web_sm', enable=["tagger", "attribute_ruler", "lemmatizer"])
        print("tokenizing with spacy")
        # Tokenize the data
        self.df['data'] = list(nlp.pipe(self.df['data'], n_process=8, batch_size=5000))
        self.df['data'] = self.df['data'].map(lambda doc: [token.text for token in doc if token.is_alpha and not token.is_stop and token.pos_ != 'PROPN'])
        #remove largest
        self.df['token_count'] = self.df['data'].map(len)
        indexes_to_remove = self.df.nlargest(10, 'token_count').index
        self.df = self.df.drop(indexes_to_remove)
        self.process_vectors()
    '''
    def preprocess(self): #same but removes stop words, numbers, proper nouns, punctuation, spaces, currency symbols etc 
        print("preprocessing and tokenizing...")
        nlp = spacy.load('en_core_web_sm', enable=["tagger", "attribute_ruler", "lemmatizer"])
        print("tokenizing with spacy")

        # Tokenize the data
        self.df['data'] = list(nlp.pipe(self.df['data'], n_process=8, batch_size=5000))

        # Updated token processing
        self.df['data'] = self.df['data'].map(lambda doc: [
            token.lemma_.lower()  # Convert to lowercase
            for token in doc
            if token.is_alpha  # Keep alphabetic tokens
            and not token.is_stop  # Remove stop words
            and not token.like_num  # Remove numbers
            and token.pos_ != 'PROPN'  # Remove proper nouns
            and not token.is_punct  # Remove punctuation
            and not token.is_space  # Remove spaces
            and not token.is_currency  # Remove currency symbols
            and token.text not in nlp.Defaults.stop_words  # Additional stop word check
        ])
        # Remove largest
        self.df['token_count'] = self.df['data'].map(len)
        indexes_to_remove = self.df.nlargest(25000, 'token_count').index
        self.df = self.df.drop(indexes_to_remove)

        self.process_vectors()

    def process_vectors(self):
        print("loading word2vec-google-news-300")
        self.data_w2v = api.load("word2vec-google-news-300")

        print("making word vectors...")
        self.not_found = 0
        def vec(word):
            if word in self.data_w2v:
                return self.data_w2v[word]
            else:
                self.not_found += 1
                return np.zeros(300)

        def doc_vec(doc):
            self.not_found = 0
            result = [vec(word) for word in doc]
            print(f"Not found {self.not_found} out of {len(doc)}")
            return result

        self.df['data'] = self.df['data'].map(doc_vec)
        
        self.df.to_pickle("../results/word_vectors.pkl")
        print("Word vector data saved")

datasetLoader = DatasetLoader()
df = datasetLoader.load("../data/IMDB_Dataset.csv")
proc = Preprocessor(df)
