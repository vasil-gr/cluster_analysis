# Cluster analysis

## Description

This library provides tools for generating and analyzing crystallization points and clusters. It includes functionalities for generating random and ideal coordinates, visualizing points, and calculating cluster densities and entropies.

## Installation

To install the library, you can use pip:

pip install git+https://github.com/vasil-gr/cluster_analysis.git

## Usage

### Generating Coordinates:

from crystal_analysis import CoordinatesGenerator
#### Initialize the CoordinatesGenerator class
coordinates = CoordinatesGenerator(box_x=100, box_y=100)

#### Generate random coordinates
random_coords = coordinates.random_spreading(N_points=50, seed=42)

#### Generate ideal coordinates
ideal_coords = coordinates.ideal_spreading(N_x_points=10)

#### Generate group of coordinates
group_coords = coordinates.add_group(50, 1600, 900)

#### Add group of coordinates to other coordinates
combined_coords = coordinates.add_group(50, 1600, 900, coords)

#### Visualize the coordinates
coordinates.show_coordinates()

### Analyzing clusters:

from crystal_analysis import ClusterAnalyzer

#### Initialize the ClusterAnalyzer class
analyzer = ClusterAnalyzer(coordinates_list=random_coords, box_x=100, box_y=100)

#### Generate cluster matrix
cluster_matrix = analyzer.generate_cluster_matrix()
print("Cluster matrix: ", cluster_matrix)

#### Calculate cluster sizes
cluster_sizes = analyzer.generate_cluster_size_list()
print("Cluster sizes: ", cluster_sizes)

#### Generate density matrix
density_matrix = analyzer.generate_density_matrix()
print("Density matrix: ", density_matrix)

#### Calculate entropy
entropy = analyzer.calculate_entropy()
print("Entropy: ", entropy)

#### Calculate_density
density = analyzer.calculate_density()
print(f"Average density: {density}")

#### Calculate coordination number
coordination_number = analyzer.calculate_coordination_number()
print(f"Сoordination number: {coordination_number}")

#### Visualize coordinates
analyzer.show_coordinates()

#### Visualize cluster map
analyzer.show_cluster_matrix()

#### Visualize matrix densities
analyzer.show_density_matrix()

## Dependencies

* numpy
* matplotlib
* opencv-python
* scipy

## Authors

* Grebenyuk Vasilii - vasya.31.46@gmail.com