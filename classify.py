from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense

class BaseModel:
    def __init__(self, input_shape, num_classes):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    def train(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

class LSTMModel(BaseModel):
    def build(self, units, num_layers=1, dropout=0.0):
        self.model = Sequential()
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # Only the last layer doesn't return sequences
            self.model.add(LSTM(units, return_sequences=return_sequences, input_shape=self.input_shape))
            if dropout > 0:
                self.model.add(Dropout(dropout))
        self.model.add(Dense(self.num_classes, activation='softmax'))


# Usage
lstm_model = LSTMModel(input_shape=(timesteps, features), num_classes=num_classes)
lstm_model.build(units=50, num_layers=2)
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.train(x_train, y_train, epochs=10, batch_size=32)
