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

	def process_vectors(self):
                print("loading word2vec-google-news-300")
                data_w2v = api.load("word2vec-google-news-300")

                print("making word vectors...")
                self.df['data'] = self.df['data'].map(lambda doc: [data_w2v[word] if word in data_w2v else np.zeros(300) for word in doc])

                self.df.to_pickle("../results/word_vectors.pkl")
                print("Word vector data saved")
datasetLoader = DatasetLoader()
df = datasetLoader.load("../data/IMDB_Dataset.csv")
proc = Preprocessor(df)
