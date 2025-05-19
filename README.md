# compBio3

This project implements a Self-Organizing Map (SOM) to cluster and visualize handwritten digits (0–9).

The input is taken from the digits_test.csv file, which contains grayscale 28×28 pixel images of digits, each represented as a flat vector of pixel values. A SOM grid (either square or hexagonal, with around 100 neurons) is trained to organize the input data so that similar digits are mapped to nearby neurons on the grid.

Each neuron becomes a representative of several similar digit samples. The final output is a visual display of the trained SOM, illustrating how the digits are grouped across the map.

Shir and Hodaya