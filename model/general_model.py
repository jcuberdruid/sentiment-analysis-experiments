from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import GRU, LSTM, SimpleRNN, Dense, Dropout, Masking, Bidirectional
from keras import mixed_precision
import numpy as np
import os
import time
import json

mixed_precision.set_global_policy('mixed_float16')

class GenModel:
    def __init__(self, input_shape, num_classes, activation_function, bidirectional=False, node_type='GRU'):
        self.model = Sequential()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.bidirectional = bidirectional
        self.node_type = node_type
        self.checkpoint_path = "best_gru_model.hdf5"

    def build(self, units, num_layers=1, dropout=0.0):
        self.num_layers = num_layers
        self.model.add(Masking(mask_value=0.0, input_shape=self.input_shape))
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            if self.node_type == 'LSTM':
                layer = LSTM(units, return_sequences=return_sequences)
            elif self.node_type == 'RNN':
                layer = SimpleRNN(units, return_sequences=return_sequences)
            else:
                layer = GRU(units, return_sequences=return_sequences)
            
            if self.bidirectional:
                layer = Bidirectional(layer)

            self.model.add(layer)
            if dropout > 0:
                self.model.add(Dropout(dropout))
        self.model.add(Dense(self.num_classes, activation=self.activation_function))

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=6)

        self.model.summary()
        start_time = time.time()
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping], validation_data=(x_val, y_val))
        training_time = time.time() - start_time

        return history, training_time

    def evaluate(self, x_test, y_test):
        self.model.load_weights(self.checkpoint_path)
        test_accuracy = self.model.evaluate(x_test, y_test)[1]
        return test_accuracy

    def save_history(self, history, test_accuracy, training_time):
        file_path=f"logs/model_info_acc_{test_accuracy}_{self.node_type}_{self.num_layers}.json"
        model_info = {
            "num_nodes": self.num_layers,
            "node_type": self.node_type,
            "bidirectional": self.bidirectional,
            "training_time": training_time,
            "test_accuracy": test_accuracy,
            "epoch_history": history.history
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file)


