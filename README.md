# Cluster analysis

## Description

This library provides tools for generating and analyzing crystallization points and clusters. It includes functionalities for generating random and ideal coordinates, visualizing points, and calculating cluster densities and entropies.

## Installation

To install the library, you can use pip:

pip install crystal_analysis

## Usage

### Generating Coordinates:

from crystal_analysis import Kords
#### Initialize the Kords class
kords = Kords(box_x=100, box_y=100)

#### Generate random coordinates
random_coords = kords.generate_random_kords(N_points=50)
print("Random Coordinates:", random_coords)

#### Generate ideal coordinates
ideal_coords = kords.generate_ideal_kords(N_x_points=10)
print("Ideal Coordinates:", ideal_coords)

### Analyzing clusters:

from crystal_analysis import Main_1

#### Initialize the Main_1 class
main = Main_1(kords_list=random_coords, box_x=100, box_y=100)

#### Visualize the coordinates
main.func_plot_kords()

#### Generate cluster matrix
cluster_matrix = main.generate_cluster_matrix(show_map=True)
print("Cluster Matrix:", cluster_matrix)

#### Calculate cluster sizes
cluster_sizes = main.generate_cluster_size_list()
print("Cluster Sizes:", cluster_sizes)

#### Generate density matrix
density_matrix = main.generate_density_matrix(show_density_map=True)
print("Density Matrix:", density_matrix)

#### Calculate entropy
entropy = main.calculate_entropy()
print("Entropy:", entropy)

## Dependencies

* numpy
* matplotlib
* opencv-python
* scipy

## Authors

* Grebenyuk Vasilii - vasya.31.46@gmail.com
* Lukiev Ivan - 456@gmail.com