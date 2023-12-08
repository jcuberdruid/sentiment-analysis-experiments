from dataset.load_pickle import LoadPickle
from model.general_model import GenModel
import numpy as np

file_path = '../results/word_vectors_trained_longer.pkl'
test_split_ratio = 0.1
retain_percentage = 100
load_pickle = LoadPickle(file_path, test_split_ratio, retain_percentage)
load_pickle.process_data()

x_train, y_train = load_pickle.train_data, load_pickle.train_labels
x_train = x_train.astype(np.float32)
x_test, y_test = load_pickle.test_data, load_pickle.test_labels
x_test = x_test.astype(np.float32)

sequence_length = load_pickle.train_data.shape[1]

node_types = ['RNN', 'LSTM', 'GRU']
num_layers_list = [1, 2, 3]
bidirectional_options = [False, True]

for node_type in node_types:
    for num_layers in num_layers_list:
        for bidirectional in bidirectional_options:
            model = GenModel(input_shape=(sequence_length, 300), num_classes=1, activation_function='sigmoid', bidirectional=bidirectional, node_type=node_type)
            model.build(units=256, num_layers=num_layers, dropout=0.2)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            history, training_time = model.train(x_train, y_train, x_test, y_test, epochs=25, batch_size=128)
            test_accuracy = model.evaluate(x_test, y_test)
            model.save_history(history, test_accuracy, training_time)
