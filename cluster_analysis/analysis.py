import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.ndimage import convolve
from typing import List, Optional, Tuple



class CoordinatesGenerator:
    """
    Класс для генерации координат точек кристаллизации.
    """

    def __init__(self, box_x: int, box_y: int) -> None:
        """
        Инициализация объекта Kords.

        Параметры:
        box_x (int): размер области по оси X.
        box_y (int): размер области по оси Y.
        """
        self.box_x = box_x
        self.box_y = box_y


    def generate_random_kords(self, N_points: int) -> List[List[int]]:
        """
        Генерация случайно распределённых координат.

        Параметры:
        N_points (int): количество генерируемых точек.

        Возвращает:
        list: список координат случайно распределённых точек.
        """
        self.random_kords = [[np.random.randint(0, self.box_x), np.random.randint(0, self.box_y)] for _ in range(N_points)]
        return self.random_kords


    def generate_ideal_kords(self, N_x_points: int) -> List[List[int]]:
        """
        Генерация идеально распределённых координат.

        Параметры:
        N_x_points (int): количество точек по оси X.

        Возвращает:
        list: список координат идеально распределённых точек.
        """
        x_values = np.linspace(self.box_x/(N_x_points*2), (N_x_points*2-1)*self.box_x/(N_x_points*2), N_x_points, dtype=int)
        d=x_values[1]-x_values[0]
        N_y_points=int(self.box_y/d)+1
        d_st=(self.box_y-(N_y_points-1)*d)/2
        y_values = np.linspace(d_st, d*(N_y_points-1)+d_st, N_y_points, dtype=int)
        X, Y = np.meshgrid(x_values, y_values)
        self.ideal_kords = np.vstack([X.ravel(), Y.ravel()]).T.tolist()
        return self.ideal_kords


    def add_group_of_kords(self, N_new_points: int, x_s: float, y_s: float, prev_cords: Optional[List[List[int]]] = None, s_d_koef: int = 12) -> List[List[int]]:
        """
        Добавление группы точек кристаллизации вокруг заданной точки с нормальным распределением.

        Параметры:
        N_new_points (int): количество новых точек.
        x_s (float): X-координата центральной точки.
        y_s (float): Y-координата центральной точки.
        prev_cords (list, optional): список существующих точек. По умолчанию - пустой список.
        s_d_koef (int, optional): коэффициент стандартного отклонения для нормального распределения. По умолчанию 12.

        Возвращает:
        list: список координат с добавленными новыми точками.
        """
        if prev_cords is None:
            prev_cords = []

        new_cords = prev_cords.copy()

        std_dev = self.box_y / s_d_koef
        points_new = []

        for _ in range(N_new_points):
            x = np.random.normal(x_s, std_dev)
            y = np.random.normal(y_s, std_dev)

            if 0 < x < self.box_x and 0 < y < self.box_y:
                points_new.append([int(x), int(y)])

        new_cords.extend(points_new)
        return new_cords




class ClusterAnalyzer:
    """
    Класс для работы с координатами точек кристаллизации.
    """

    def __init__(self, kords_list: List[List[int]], box_x: int, box_y: int, r_cut: float = 1.618) -> None:
        """
        Инициализация объекта Main_1.

        Параметры:
        kords_list (list): список координат точек.
        box_x (int): размер области по оси X.
        box_y (int): размер области по оси Y.
        r_cut (float, optional): параметр, определяющий предельный радиус кластера. По умолчанию 1.618.
        """
        self.kords_list = kords_list
        self.box_x = box_x
        self.box_y = box_y
        self.r_cut = r_cut
        self.cluster_matrix: Optional[np.ndarray] = None
        self.cluster_sizes: Optional[List[int]] = None
        self.density_matrix: Optional[np.ndarray] = None


    def func_plot_kords(self) -> None:
        """
        Визуализация координат точек.
        """
        img_points_koord = np.zeros((self.box_y, self.box_x), dtype=np.uint8)
        for x, y in self.kords_list:
            cv2.circle(img_points_koord, (x, y), radius=4, color=(255, 255, 255), thickness=-1)

        plt.imshow(img_points_koord, cmap='gray')
        plt.axis('off')


    def generate_cluster_matrix(self, show_map: bool = False) -> np.ndarray:
        """
        Генерация и возвращение размеченной матрицы кластеров.

        Параметры:
        show_map (bool, optional): если True, генерирует карту кластеров. По умолчанию False.

        Возвращает:
        np.ndarray: матрица кластеров.
        """

        num_points = len(self.kords_list)
        radius = np.sqrt(self.box_x * self.box_y / num_points) * self.r_cut # предельный радиус кластера
        self.cluster_matrix = np.ones((self.box_y, self.box_x), dtype=int) * -1 # матрица для хранения индексов кластеров

        # функция обработки блока данных (чтобы хватало памяти)
        def process_block(x_start: int, x_end: int, y_start: int, y_end: int) -> np.ndarray:
            x_indices, y_indices = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
            coords = np.c_[x_indices.ravel(), y_indices.ravel()]
            dists = distance.cdist(coords, self.kords_list, 'sqeuclidean')
            min_dists = np.min(dists, axis=1)
            min_indices = np.argmin(dists, axis=1)
            mask = min_dists < radius**2
            min_indices[~mask] = -1
            block_cluster_matrix = np.ones((y_end - y_start, x_end - x_start), dtype=int) * -1
            block_cluster_matrix[y_indices.ravel() - y_start, x_indices.ravel() - x_start] = min_indices
            return block_cluster_matrix

        block_size = 100 # размер блока
        for x_start in range(0, self.box_x, block_size): # обработка по блокам
            for y_start in range(0, self.box_y, block_size):
                x_end = min(x_start + block_size, self.box_x)
                y_end = min(y_start + block_size, self.box_y)
                self.cluster_matrix[y_start:y_end, x_start:x_end] = process_block(x_start, x_end, y_start, y_end)

        # визуализация
        if show_map:
            data = np.ones((self.box_y, self.box_x, 3), dtype=int) * 255
            colors = np.random.randint(256, size=(len(self.kords_list), 3), dtype=int)
            for y in range(self.box_y):
                for x in range(self.box_x):
                    if self.cluster_matrix[y, x] != -1:
                        data[y, x, :] = colors[self.cluster_matrix[y, x]]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(data)
            for point in self.kords_list:
                ax.plot(point[0], point[1], 'o', color='black')
            ax.set_title('Карта кластеров')
            plt.show()

        return self.cluster_matrix


    def generate_cluster_size_list(self) -> List[int]:
        """
        Генерация списка размеров кластеров.

        Возвращает:
        list: список размеров кластеров (количество пикселей в каждом кластере).
        """

        if self.cluster_matrix is None:
            raise ValueError("Сперва необходимо сгенерировать матрицу кластеров. Для этого используйте функцию 'generate_cluster_matrix'!")
        num_clusters = len(self.kords_list) # список для хранения размеров кластеров
        self.cluster_sizes = [0] * (num_clusters + 1)
        for index in range(num_clusters): # подсчет количества пикселей для каждого кластера
            self.cluster_sizes[index] = np.sum(self.cluster_matrix == index)
        self.cluster_sizes[-1] = np.sum(self.cluster_matrix == -1) # количество пикселей, не отнесенных ни к одному кластеру
        return self.cluster_sizes


    def generate_density_matrix(self, smo_num: int = 0, show_density_map: bool = False) -> np.ndarray:
        """
        Генерация и возвращение матрицы плотности кластеров.

        Параметры:
        smo_num (int, optional): количество сглаживаний матрицы плотности. По умолчанию 0.
        show_density_map (bool, optional): если True, отображает карту плотности. По умолчанию False.

        Возвращает:
        np.ndarray: матрица плотности кластеров.
        """
        if self.cluster_matrix is None or self.cluster_sizes is None:
            raise ValueError("Необходимо сперва сгенерировать матрицу кластеров и список размеров кластеров!")

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

        # !!!!!!!!!!!!!! Нужно доработать
        # if smo_num > 0:
        #     kernel = np.array([[1/8, 1/8, 1/8], [1/8, 0, 1/8], [1/8, 1/8, 1/8]])
        #     for _ in range(smo_num):
        #         density_sum_before = self.density_matrix.sum()
        #         d_matrix = convolve(self.density_matrix, kernel, mode='constant', cval=0.0)
        #         density_sum_after = d_matrix.sum()
        #         if density_sum_after != 0:
        #             self.density_matrix *= density_sum_before / density_sum_after  # нормализуем матрицу плотности


        if show_density_map:
            plt.imshow(self.density_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Density')
            plt.title('Карта плотности кластеров')
            plt.show()

        return self.density_matrix


    def calculate_entropy(self) -> float:
        """
        Вычисление энтропии матрицы плотности.

        Возвращает:
        float: Значение энтропии.
        """
        if self.density_matrix is None:
            raise ValueError("Необходимо сперва сгенерировать матрицу плотности. Для этого используйте функцию 'generate_density_matrix'!")

        density_matrix_flat = self.density_matrix.flatten() # выпрямление в вектор
        non_zero_densities = density_matrix_flat[density_matrix_flat > 0]

        entropy = -np.sum(non_zero_densities * np.log(non_zero_densities))

        N = len(non_zero_densities)
        c_mean = np.mean(non_zero_densities)
        entropy += N * c_mean * np.log(c_mean)

        return entropy