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

	def preprocess(self):  # same but removes stop words, numbers, proper nouns, punctuation, spaces, currency symbols etc 
		print("preprocessing and tokenizing...")
		nlp = spacy.load('en_core_web_sm', enable=["tagger", "attribute_ruler", "lemmatizer"])
		print("tokenizing with spacy")

		# Tokenize the data
		self.df['data'] = list(nlp.pipe(self.df['data'], n_process=8, batch_size=5000))

		# Updated token processing
		self.df['data'] = self.df['data'].map(lambda doc: [
			token.lemma_.lower()  # Convert to lemmatized lowercase
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
