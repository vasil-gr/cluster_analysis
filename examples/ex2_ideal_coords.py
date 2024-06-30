# Example of using the library to work with ideal coordinates
from cluster_analysis import CoordinatesGenerator, ClusterAnalyzer

# Initialize the CoordinatesGenerator class with area size 2000x1400
coordinates = CoordinatesGenerator(2000, 1400)
# Generate ideally distributed coordinates (12 points along the X axis)
ideal_coords = coordinates.ideal_spreading(12)
# Visualize the coordinates
coordinates.show_coordinates()

# Initialize the ClusterAnalyzer class with the obtained ideal coordinates
analyzer = ClusterAnalyzer(ideal_coords, 2000, 1400, r_cut=1)
# Generate the cluster matrix
cluster_matrix = analyzer.generate_cluster_matrix()
# Get the list of cluster sizes
cluster_size_list = analyzer.generate_cluster_size_list()
# Generatethe density matrix (without smoothing)
density_matrix = analyzer.generate_density_matrix()
# Generatethe density matrix (with smoothing)
# density_matrix_sm10 = analyzer.generate_density_matrix(smo_num=10)

# Calculate entropy
entropy = analyzer.calculate_entropy()
print(f"Entropy: {entropy}")
# Calculate average density
density = analyzer.calculate_density()
print(f"Average density: {density}")

# Visualize coordinates, cluster matrix, density matrix
# analyzer.show_coordinates()
analyzer.show_cluster_matrix()
analyzer.show_density_matrix()