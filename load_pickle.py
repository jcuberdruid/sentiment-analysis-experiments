import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class LoadPickle:
	def __init__(self, file_path, test_split_ratio, retain_percentage=100):
		self.df = pd.read_pickle(file_path)
		largest = 0;	
		if retain_percentage < 100:
			self.df = self.df.sample(frac=retain_percentage / 100)
		self.train_test_split_ratio = test_split_ratio

	def process_data(self):
		sequences = self.df.iloc[:, 0].tolist()
		text_labels = self.df.iloc[:, 1].tolist()
		self.labels = [0 if x == 'negative' else 1 for x in text_labels]

		print("##########################################")
		print("Padding: ")
		padded_sequences = pad_sequences(sequences, padding='post')
		sequences_array = np.array(padded_sequences)
		labels_array = np.array(self.labels)
		#split labels
		self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
		sequences_array, labels_array, test_size=self.train_test_split_ratio)

		print(f"train data: {self.train_data.shape}")
		print(f"train labels: {self.train_labels.shape}")
		print(f"test data: {self.test_data.shape}")
		print(f"test labels: {self.test_labels.shape}")
#load_pickle = LoadPickle('../results/word_vectors.pkl', test_split_ratio=0.2, retain_percentage=50)
#load_pickle.process_data()
