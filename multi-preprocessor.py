import numpy as np
import spacy
import gensim.downloader as api
from multiprocessing import Pool
from data_handler import Data_handler

def tokenizer_init():
	global nlp
	nlp = spacy.load('en_core_web_sm')

def tokenize(string):
	doc = nlp(string.lower())
	return [token.text for token in doc if token.is_alpha and not token.is_stop]

class Preprocessor:
	def __init__(self, data):
		self.processed_data = []
		self.preprocess(data)

	def preprocess(self, data):
		print("preprocessing and tokenizing...")
		with Pool(14, initializer=tokenizer_init) as p:
			tokenized_data = p.map(tokenize, data)

		# Load the model just once
		self.data_w2v = api.load("word2vec-google-news-300")

		print("making word vectors...")
		self.process_vectors(tokenized_data)

	def process_vectors(self, tokenized_data):
		vectorized_data = []
		for doc in tokenized_data:
			print([word in doc][:12])
			doc_vector = [self.data_w2v[word] if word in self.data_w2v else np.zeros(300) for word in doc]
			vectorized_data.append(doc_vector)

		max_len = max(len(doc) for doc in vectorized_data)
		padded_data = np.array([np.pad(doc, ((0, max_len - len(doc)), (0, 0)), mode='constant') for doc in vectorized_data])

		np.save('../results/word_vectors.npy', padded_data)
		print("Word vector data saved")

load = Data_handler("../data/IMDB_Dataset.csv")
proc = Preprocessor(load.data)

