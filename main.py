import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('digits_test.csv', header=None)
labels = pd.read_csv('digits_keys.csv', header=None)

# Normalize the data to be in the range [0, 1]
data = data / 255.0

# Step 2: Define the SOM grid
grid_size = (10, 10)  # 10x10 grid
n_neurons = grid_size[0] * grid_size[1]  # Total number of neurons
input_len = data.shape[1]  # Length of input vector

# Step 3: Initialize the SOM with random values
som = np.random.random((n_neurons, input_len))


def find_bmu(som, input_vector):
    """ Find the Best Matching Unit (BMU) for a given input vector """
    distances = cdist(som, input_vector.reshape(1, -1))
    bmu_index = np.argmin(distances)
    return bmu_index


def find_second_bmu(som, input_vector, bmu_index):
    """ Find the second Best Matching Unit (second BMU) for a given input vector """
    distances = cdist(som, input_vector.reshape(1, -1))
    distances[bmu_index] = np.inf  # Exclude the BMU
    second_bmu_index = np.argmin(distances)
    return second_bmu_index


def update_weights(som, bmu_index, input_vector, learning_rate, radius):
    """ Update the weights of the neurons """
    for i in range(len(som)):
        distance = np.linalg.norm(np.array([i // grid_size[1], i % grid_size[1]]) - np.array(
            [bmu_index // grid_size[1], bmu_index % grid_size[1]]))
        if distance < radius:
            influence = np.exp(-distance ** 2 / (2 * (radius ** 2)))
            som[i] += learning_rate * influence * (input_vector - som[i])


# Training parameters
n_iterations = 10  # Number of iterations
learning_rate = 0.1  # Initial learning rate
initial_radius = max(grid_size) / 2  # Initial radius
time_constant = n_iterations / np.log(initial_radius)

# Step 4: Training the SOM
topological_errors = 0
quantization_errors = 0

for iteration in range(n_iterations):
    for input_vector in data.values:
        bmu_index = find_bmu(som, input_vector)
        second_bmu_index = find_second_bmu(som, input_vector, bmu_index)

        # Calculate topological error
        bmu_coords = np.array([bmu_index // grid_size[1], bmu_index % grid_size[1]])
        second_bmu_coords = np.array([second_bmu_index // grid_size[1], second_bmu_index % grid_size[1]])
        if np.linalg.norm(bmu_coords - second_bmu_coords) > 1:
            topological_errors += 1

        # Calculate quantization error
        quantization_errors += np.linalg.norm(input_vector - som[bmu_index])

        radius = initial_radius * np.exp(-iteration / time_constant)
        current_learning_rate = learning_rate * np.exp(-iteration / n_iterations)
        update_weights(som, bmu_index, input_vector, current_learning_rate, radius)

# Normalize errors
topological_error = topological_errors / (n_iterations * len(data))
quantization_error = quantization_errors / (n_iterations * len(data))

# Step 5: Evaluate and visualize the results
neuron_counts = np.zeros((grid_size[0], grid_size[1], 10))  # To count the digits in each neuron

for i, input_vector in enumerate(data.values):
    bmu_index = find_bmu(som, input_vector)
    neuron_counts[bmu_index // grid_size[1], bmu_index % grid_size[1], labels.iloc[i, 0]] += 1

# Visualize the SOM
fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
fig.suptitle(f'Topological Error: {topological_error:.4f}, Quantization Error: {quantization_error:.4f}', fontsize=16)

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        dominant_digit = np.argmax(neuron_counts[i, j])
        percentage = (neuron_counts[i, j, dominant_digit] / np.sum(neuron_counts[i, j])) * 100 if np.sum(
            neuron_counts[i, j]) > 0 else 0
        ax[i, j].imshow(som[i * grid_size[1] + j].reshape(28, 28), cmap='gray')
        ax[i, j].set_title(f'{dominant_digit} ({percentage:.1f}%)', fontsize=8)
        ax[i, j].axis('off')

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Add more space between subplots
plt.show()
