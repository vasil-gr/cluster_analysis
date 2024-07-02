# Example of using the library to work with a group of coordinates around a given point
from cluster_analysis import CoordinatesGenerator, ClusterAnalyzer

# Initialize the CoordinatesGenerator class with area size 2000x1400
coordinates = CoordinatesGenerator(2000, 1400)
# Generate a group of coordinates around the point (1600, 900)
group_coords = coordinates.add_group(50, 1600, 900, seed=42)
# Visualize the coordinates
coordinates.show_coordinates()

# Initialize the ClusterAnalyzer class with the obtained group coordinates
analyzer = ClusterAnalyzer(group_coords, 2000, 1400, r_cut=1)
# Generate the cluster matrix
cluster_matrix = analyzer.generate_cluster_matrix()
# Get the list of cluster sizes
cluster_size_list = analyzer.generate_cluster_size_list()
# Generatethe density matrix (without smoothing)
density_matrix = analyzer.generate_density_matrix()

# Calculate entropy
entropy = analyzer.calculate_entropy()
print(f"Entropy: {entropy}")
# Calculate average density
density = analyzer.calculate_density()
print(f"Average density: {density}")
# Calculate coordination number
coordination_number = analyzer.calculate_coordination_number()
print(f"Сoordination number: {coordination_number}")

# Visualize coordinates, cluster matrix, density matrix
# analyzer.show_coordinates()
analyzer.show_cluster_matrix()
analyzer.show_density_matrix()