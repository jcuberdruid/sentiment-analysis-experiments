import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class LoadPickle:
	def __init__(self, filepath, train_test_split_ratio):
		self.df = pd.read_pickle(filepath)
		self.train_test_split_ratio = train_test_split_ratio
		self.process_data()

	def process_data(self):
		sequences = self.df.iloc[:, 0].tolist()
		padded_sequences = pad_sequences(sequences, padding='post')
		sequences_array = np.array(padded_sequences)
		self.train_data, self.test_data = train_test_split(sequences_array, test_size=self.train_test_split_ratio)
		print(self.train_data.shape)
		print(self.test.shape)

load_pickle = LoadPickle('../results/word_vectors.pkl', 0.2)

