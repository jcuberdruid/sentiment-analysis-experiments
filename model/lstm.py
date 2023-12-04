from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras import mixed_precision
import numpy as np

mixed_precision.set_global_policy('mixed_float16')

# LSTM Model
class LSTMModel:
    def __init__(self, input_shape, num_classes, activation_function):
        self.model = Sequential()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.checkpoint_path = "best_lstm_model.hdf5"  # Path to save the best model

    def build(self, units, num_layers=1, dropout=0.0):
        self.model.add(Masking(mask_value=0.0, input_shape=self.input_shape))
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            if i == 0:
                self.model.add(LSTM(units, return_sequences=return_sequences, input_shape=self.input_shape))
            else:
                self.model.add(LSTM(units, return_sequences=return_sequences))
            if dropout > 0:
                self.model.add(Dropout(dropout))
        self.model.add(Dense(self.num_classes, activation=self.activation_function))

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        # Callbacks for early stopping and saving the best model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        self.model.summary()
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint], validation_data=(x_val, y_val))

    def evaluate(self, x_test, y_test):
        # Load the best model
        self.model.load_weights(self.checkpoint_path)
        return self.model.evaluate(x_test, y_test)
