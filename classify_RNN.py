from load_pickle import LoadPickle
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Masking
from keras.utils import to_categorical
from keras import mixed_precision
import numpy as np

mixed_precision.set_global_policy('mixed_float16')

# Load data
file_path = '../results/word_vectors_trained.pkl'
test_split_ratio = 0.2  # Modify as needed
retain_percentage = 100 # Modify as needed
load_pickle = LoadPickle(file_path, test_split_ratio, retain_percentage)
load_pickle.process_data()

# RNN Model
class RNNModel:
	def __init__(self, input_shape, num_classes):
		self.model = Sequential()
		self.input_shape = input_shape
		self.num_classes = num_classes

	def build(self, units, num_layers=1, dropout=0.0):
		self.model.add(Masking(mask_value=0.0, input_shape=self.input_shape))
		for i in range(num_layers):
			return_sequences = i < num_layers - 1  
			if i == 0:
				self.model.add(SimpleRNN(units, return_sequences=return_sequences, input_shape=self.input_shape))
			else:
				self.model.add(SimpleRNN(units, return_sequences=return_sequences))
			if dropout > 0:
				self.model.add(Dropout(dropout))
		self.model.add(Dense(self.num_classes, activation='sigmoid'))

	def compile(self, optimizer, loss, metrics):
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	def train(self, x_train, y_train, epochs, batch_size):
		self.model.summary()
		self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

	def evaluate(self, x_test, y_test):
		return self.model.evaluate(x_test, y_test)

# Prepare data
x_train, y_train = load_pickle.train_data, load_pickle.train_labels
x_train = x_train.astype(np.float32)
x_test, y_test = load_pickle.test_data, load_pickle.test_labels
x_test = x_test.astype(np.float32)

# After loading and processing the data
sequence_length = load_pickle.train_data.shape[1]  # Dynamically get the sequence length

# Instantiate and build the RNN model with the dynamic sequence length
rnn_model = RNNModel(input_shape=(sequence_length, 300), num_classes=1)
rnn_model.build(units=256, num_layers=1, dropout=0.2)
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.train(load_pickle.train_data, load_pickle.train_labels, epochs=100, batch_size=64)
rnn_model.evaluate(load_pickle.test_data, load_pickle.test_labels)

