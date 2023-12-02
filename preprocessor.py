import spacy
from data_handler import Data_handler

class Preprocessor:
	nlp = spacy.load('en_core_web_sm')
	processed_data = []
	def __init__(self, data):
		self.preprocess(data)
	
	def tokenize(self, string): #already removed breaklines with sed
		doc = self.nlp(string.lower())
		return [token.text for token in doc if token.is_alpha]  # excludes punctuation
		
	def preprocess(self, data):
		for x in data:
			self.processed_data.append(self.tokenize(x))

load = Data_handler("../data/IMDB_Dataset.csv")
proc = Preprocessor(load.data)

print(proc.processed_data[0])

