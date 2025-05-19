import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
import os
import sys

# Utility function to handle paths
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Start the timer
start_time = time.time()

# Load the data
data = pd.read_csv(resource_path('digits_test.csv'), header=None)
labels = pd.read_csv(resource_path('digits_keys.csv'), header=None)

# Normalize the data to be in the range [0, 1]
data = data / 255.0

# Define the SOM grid
grid_size = (10, 10)  # 10x10 grid
n_neurons = grid_size[0] * grid_size[1]  # Total number of neurons
input_len = data.shape[1]  # Length of input vector

# Initialize the SOM with random values from a Gaussian distribution
mean = data.mean().mean()
std = data.std().std()
som = np.random.normal(loc=mean, scale=std, size=(n_neurons, input_len))

# Find the Best Matching Unit (BMU) for a given input vector
def find_bmu(som, input_vector):
    distances = cdist(som, input_vector.reshape(1, -1))
    bmu_index = np.argmin(distances)
    return bmu_index

# Find the second Best Matching Unit (second BMU) for a given input vector
def find_second_bmu(som, input_vector, bmu_index):
    distances = cdist(som, input_vector.reshape(1, -1))
    distances[bmu_index] = np.inf  # Exclude the BMU
    second_bmu_index = np.argmin(distances)
    return second_bmu_index

# performing the update rule
# Update the weights of the neurons
def update_weights(som, bmu_index, input_vector, learning_rate, radius):
    for i in range(len(som)):
        distance = np.linalg.norm(np.array([i // grid_size[1], i % grid_size[1]]) - np.array([bmu_index // grid_size[1], bmu_index % grid_size[1]]))
        if distance < radius:
            neighborhood = np.exp(-distance ** 2 / (2 * (radius ** 2)))
            som[i] += learning_rate * neighborhood * (input_vector - som[i])

# Training parameters
n_iterations = 10  # Number of iterations
learning_rate = 0.1  # Initial learning rate
initial_radius = max(grid_size) / 2  # Initial radius
time_constant = n_iterations / np.log(initial_radius)

# Training the SOM
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

# End the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60

# Evaluate and visualize the results
neuron_counts = np.zeros((grid_size[0], grid_size[1], 10))  # To count the digits in each neuron

for i, input_vector in enumerate(data.values):
    bmu_index = find_bmu(som, input_vector)
    neuron_counts[bmu_index // grid_size[1], bmu_index % grid_size[1], labels.iloc[i, 0]] += 1

# Visualize the SOM
fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
fig.suptitle(f'Topological Error: {topological_error:.4f}, Quantization Error: {quantization_error:.4f}\nElapsed Time: {elapsed_time_minutes:.2f} minutes', fontsize=16)

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        dominant_digit = np.argmax(neuron_counts[i, j])
        percentage = (neuron_counts[i, j, dominant_digit] / np.sum(neuron_counts[i, j])) * 100 if np.sum(neuron_counts[i, j]) > 0 else 0
        ax[i, j].imshow(som[i * grid_size[1] + j].reshape(28, 28), cmap='gray')
        ax[i, j].set_title(f'{dominant_digit} ({percentage:.1f}%)', fontsize=12)
        ax[i, j].axis('off')
# Add more space between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)  
 # Display the plot
plt.show() 
