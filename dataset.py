import pandas as pd
import csv

class Record:
	def __init__(self, label, data):
		self.label = label
		self.data = data

class DatasetLoader:
	def load(self, filepath):
		dataset = pd.DataFrame(columns=['data', 'sentiment'])
		with open(filepath) as csvfile:
			reader = csv.reader(csvfile, delimiter=',')	
			next(reader)
			dataset = pd.DataFrame(data=reader, columns=['data', 'label'])
			return dataset
