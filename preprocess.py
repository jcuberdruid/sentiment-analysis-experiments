import spacy
import numpy as np
import pandas as pd
import gensim.downloader as api
from multiprocessing import Pool
from dataset import DatasetLoader
from gensim.models import Word2Vec

class Preprocessor:
	def __init__(self, df: pd.DataFrame):
		self.df = df
		self.preprocess()

	def preprocess(self):  
		print("preprocessing and tokenizing...")
		nlp = spacy.load('en_core_web_sm', enable=["tagger", "attribute_ruler", "lemmatizer"])
		print("tokenizing with spacy")

		self.df['data'] = list(nlp.pipe(self.df['data'], n_process=8, batch_size=5000))

		self.df['data'] = self.df['data'].map(lambda doc: [
			token.lemma_.lower()  
			for token in doc
			if token.is_alpha  
			and not token.is_stop  
			and not token.like_num 
			and token.pos_ != 'PROPN'
			and not token.is_punct  
			and not token.is_space 
			and not token.is_currency 
			and token.text not in nlp.Defaults.stop_words  
		])
		self.df['token_count'] = self.df['data'].map(len)
		indexes_to_remove = self.df.nlargest(20000, 'token_count').index
		self.df = self.df.drop(indexes_to_remove)

		self.process_vectors()

	def process_vectors(self):
		print("Training custom Word2Vec model...")
		data_w2v = Word2Vec(self.df['data'], vector_size=300, window=5, min_count=1, workers=4)
		print("Making word vectors...")
		self.df['data'] = self.df['data'].map(lambda doc: [data_w2v.wv[word] if word in data_w2v.wv else np.zeroes(300) for word in doc])
		self.df.to_pickle("../results/word_vectors_trained_longer.pkl")
		print("Word vector data saved")


datasetLoader = DatasetLoader()
df = datasetLoader.load("../data/IMDB_Dataset.csv")
proc = Preprocessor(df)
