from dataset.load_pickle import LoadPickle
from model.rnn import GenModel  # Make sure this points to the updated class
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

activation_functions = ['sigmoid', 'relu', 'tanh']

for activation in activation_functions:
    rnn_model = GenModel(input_shape=(sequence_length, 300), num_classes=1, activation_function=activation)
    rnn_model.build(units=256, num_layers=1, dropout=0.2)
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    rnn_model.train(
        load_pickle.train_data,
        load_pickle.train_labels,
        load_pickle.test_data,
        load_pickle.test_labels,
        epochs=25,
        batch_size=128
    )
    evaluation_results = rnn_model.evaluate(load_pickle.test_data, load_pickle.test_labels)
    print(f"Using activation function {activation}, eval loss, eval accuracy:", evaluation_results)
