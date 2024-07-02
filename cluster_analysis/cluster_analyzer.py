import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.ndimage import convolve
from typing import List, Optional



class ClusterAnalyzer:
    """
    Class for working with coordinates of crystallization point 
    """

    def __init__(self, coordinates_list: List[List[int]], box_x: int, box_y: int, r_cut: float = 1.618) -> None:
        """
        Initialize the ClusterAnalyzer object.

        Parameters:
        coordinates_list (list): list of point coordinates.
        box_x (int): size of the area along the X axis.
        box_y (int): size of the area along the Y axis.
        r_cut (float, optional): parameter defining the cluster radius. Default is 1.618.
        """
        self.coordinates_list = coordinates_list
        self.box_x = box_x
        self.box_y = box_y
        self.r_cut = r_cut
        self.cluster_matrix: Optional[np.ndarray] = None
        self.cluster_sizes: Optional[List[int]] = None
        self.density_matrix: Optional[np.ndarray] = None


    def generate_cluster_matrix(self) -> np.ndarray:
        """
        Generate and return the cluster matrix.

        Returns:
        np.ndarray: cluster matrix.
        """

        num_points = len(self.coordinates_list)
        radius = np.sqrt(self.box_x * self.box_y / num_points) * self.r_cut # cluster radius
        self.cluster_matrix = np.ones((self.box_y, self.box_x), dtype=int) * -1 # matrix for storing cluster indices

        # data block processing function (to ensure enough memory)
        def process_block(x_start: int, x_end: int, y_start: int, y_end: int) -> np.ndarray:
            x_indices, y_indices = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
            coords = np.c_[x_indices.ravel(), y_indices.ravel()]
            dists = distance.cdist(coords, self.coordinates_list, 'sqeuclidean')
            min_dists = np.min(dists, axis=1)
            min_indices = np.argmin(dists, axis=1)
            mask = min_dists < radius**2
            min_indices[~mask] = -1
            block_cluster_matrix = np.ones((y_end - y_start, x_end - x_start), dtype=int) * -1
            block_cluster_matrix[y_indices.ravel() - y_start, x_indices.ravel() - x_start] = min_indices
            return block_cluster_matrix

        block_size = 100 # block size
        for x_start in range(0, self.box_x, block_size): # process by blocks
            for y_start in range(0, self.box_y, block_size):
                x_end = min(x_start + block_size, self.box_x)
                y_end = min(y_start + block_size, self.box_y)
                self.cluster_matrix[y_start:y_end, x_start:x_end] = process_block(x_start, x_end, y_start, y_end)

        return self.cluster_matrix


    def generate_cluster_size_list(self) -> List[int]:
        """
        Generate a list of cluster sizes.

        Returns:
        list: list of cluster sizes (number of pixels in each cluster).
        """

        if self.cluster_matrix is None:
            raise ValueError("First, generate the cluster matrix using the 'generate_cluster_matrix' method!")
        num_clusters = len(self.coordinates_list) # list for storing cluster sizes
        self.cluster_sizes = [0] * (num_clusters + 1)
        for index in range(num_clusters): # count the number of pixels for each cluster
            self.cluster_sizes[index] = np.sum(self.cluster_matrix == index)
        self.cluster_sizes[-1] = np.sum(self.cluster_matrix == -1) # number of pixels not assigned to any cluster
        return self.cluster_sizes


    def generate_density_matrix(self, smo_num: int = 0) -> np.ndarray:
        """
        Generate the density matrix.

        Parameters:
        smo_num (int, optional): number of density matrix smoothings. Default is 0.

        Returns:
        np.ndarray: density matrix.
        """
        if self.cluster_matrix is None or self.cluster_sizes is None:
            raise ValueError("First, generate the cluster matrix and cluster size list!")

        self.density_matrix = np.zeros((self.box_y, self.box_x), dtype=float)
        total_pixels = self.box_x * self.box_y
        n_no_cluster = self.cluster_sizes[-1]

        for y in range(self.box_y):
            for x in range(self.box_x):
                cluster_index = self.cluster_matrix[y, x]
                if cluster_index != -1:
                    n_cluster = self.cluster_sizes[cluster_index]
                    density = n_cluster / (total_pixels - n_no_cluster)
                    self.density_matrix[y, x] = density

        # !!!!!!!!!!!!!! Needs improvement
        # if smo_num > 0:
        #     kernel = np.array([[1/8, 1/8, 1/8], [1/8, 0, 1/8], [1/8, 1/8, 1/8]])
        #     for _ in range(smo_num):
        #         density_sum_before = self.density_matrix.sum()
        #         d_matrix = convolve(self.density_matrix, kernel, mode='constant', cval=0.0)
        #         density_sum_after = d_matrix.sum()
        #         if density_sum_after != 0:
        #             self.density_matrix *= density_sum_before / density_sum_after  # normalize the density matrix


        return self.density_matrix


    def calculate_entropy(self) -> float:
        """
        Calculate the entropy of the density matrix.

        Returns:
        float: Entropy value.
        """
        if self.density_matrix is None:
            raise ValueError("First, generate the density matrix using the 'generate_density_matrix' method!")

        density_matrix_flat = self.density_matrix.flatten() # flatten to vector
        non_zero_densities = density_matrix_flat[density_matrix_flat > 0]

        entropy = -np.sum(non_zero_densities * np.log(non_zero_densities))

        N = len(non_zero_densities)
        c_mean = np.mean(non_zero_densities)
        entropy += N * c_mean * np.log(c_mean)

        return entropy
    
    
    def calculate_density(self) -> float:
        """
        Calculate average density of points.

        Returns:
        float: Density value (number of points per pixel area).
        """
        total_points = len(self.coordinates_list)
        total_area = self.box_x * self.box_y
        density = total_points / total_area
        return density
    
    
    def calculate_coordination_number(self) -> float:
        """
        Calculate coordination number of points.

        Returns:
        float: Coordination number value (average number of points surrounding a point).
        """
        total_points = len(self.coordinates_list)
        if total_points == 0:
            return 0
        
        concentration = self.box_x * self.box_y / total_points
        radius = concentration ** 0.5
        
        points_array = np.array(self.coordinates_list)
        distances = distance.cdist(points_array, points_array, 'euclidean')
        neighbors_count = np.sum(distances <= radius, axis=1) - 1
        coordination_number = neighbors_count.mean()
        return coordination_number





    def show_coordinates(self) -> None:
        """
        Visualization of point coordinates.
        """
        img_points_koord = np.zeros((self.box_y, self.box_x), dtype=np.uint8)
        for x, y in self.coordinates_list:
            cv2.circle(img_points_koord, (x, y), radius=4, color=(255, 255, 255), thickness=-1)

        plt.imshow(img_points_koord, cmap='gray')
        plt.axis('off')
        plt.show()
    
    
    def show_cluster_matrix(self) -> None:
        """
        Visualization of cluster matrix.
        """
        if self.cluster_matrix is None:
            raise ValueError("First, generate the cluster matrix using the 'generate_cluster_matrix' method!")
        
        data = np.ones((self.box_y, self.box_x, 3), dtype=int) * 255
        colors = np.random.randint(256, size=(len(self.coordinates_list), 3), dtype=int)
        for y in range(self.box_y):
            for x in range(self.box_x):
                if self.cluster_matrix[y, x] != -1:
                    data[y, x, :] = colors[self.cluster_matrix[y, x]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(data)
        for point in self.coordinates_list:
            ax.plot(point[0], point[1], 'o', color='black')
        ax.set_title('Карта кластеров')
        plt.show()
    
    
    def show_density_matrix(self) -> None:
        """
        Visualization of density matrix.
        """
        if self.density_matrix is None:
            raise ValueError("First, generate the density matrix using the 'generate_density_matrix' method!")
        
        plt.imshow(self.density_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Density')
        plt.title('Карта плотности кластеров')
        plt.show()
        