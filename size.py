
import numpy as np

# Load the .npy file
file_path = '../results/word_vectors_batch_20000.npy'
word_vectors = np.load(file_path)

# Print the shape of the loaded array
print(word_vectors.shape)
