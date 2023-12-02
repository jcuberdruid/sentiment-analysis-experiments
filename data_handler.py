import numpy as np 
import csv

class Data_handler:
	def __init__(self, filepath):
		self.loadData(filepath)
	
	def loadData(self, filepath):
		self.data = []
		self.labels = []
		with open(filepath) as csvfile:
			reader = csv.reader(csvfile, delimiter=',')	
			next(reader)
			for row in reader:
				self.data.append(row[0])
				self.labels.append(row[0])

#bla = Data_handler("../data/IMDB_Dataset.csv")
#print(bla.data[0])
