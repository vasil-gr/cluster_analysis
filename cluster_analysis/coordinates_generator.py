import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional



class CoordinatesGenerator:
    """
    Класс для генерации координат точек кристаллизации.
    Class for generating coordinates of crystallization point
    """

    def __init__(self, box_x: int, box_y: int) -> None:
        """
        Initialize the CoordinatesGenerator object.

        Параметры:
        box_x (int): size of the area along the X axis.
        box_y (int): size of the area along the Y axis.
        """
        self.box_x = box_x
        self.box_y = box_y


    def random_spreading(self, N_points: int, seed: Optional[int] = None) -> List[List[int]]:
        """
        Generate random coordinates.

        Parameters:
        N_points (int): number of generated points.
        seed (Optional[int]): seed value for the random number generator.

        Returns:
        list: list of random point coordinates.
        """
        if seed is not None:
            np.random.seed(seed)
        
        
        self.coordinates = [[np.random.randint(0, self.box_x), np.random.randint(0, self.box_y)] for _ in range(N_points)]
        return self.coordinates


    def ideal_spreading(self, N_x_points: int) -> List[List[int]]:
        """
        Generate ideal coordinates.

        Parameters:
        N_x_points (int): number of points along the X axis.

        Returns:
        list: list of ideal point coordinates.
        """
        x_values = np.linspace(self.box_x/(N_x_points*2), (N_x_points*2-1)*self.box_x/(N_x_points*2), N_x_points, dtype=int)
        d=x_values[1]-x_values[0]
        N_y_points=int(self.box_y/d)+1
        d_st=(self.box_y-(N_y_points-1)*d)/2
        y_values = np.linspace(d_st, d*(N_y_points-1)+d_st, N_y_points, dtype=int)
        X, Y = np.meshgrid(x_values, y_values)
        self.coordinates = np.vstack([X.ravel(), Y.ravel()]).T.tolist()
        return self.coordinates


    def add_group(self, N_new_points: int, x_s: float, y_s: float, prev_cords: Optional[List[List[int]]] = None, s_d_koef: int = 12, seed: Optional[int] = None) -> List[List[int]]:
        """
        Add a group of crystallization points around a given point with normal distribution.

        Parameters:
        N_new_points (int): number of new points.
        x_s (float): X-coordinate of the central point.
        y_s (float): Y-coordinate of the central point.
        prev_cords (list, optional): list of existing points. Default is an empty list.
        s_d_koef (int, optional): standard deviation coefficient for normal distribution. Default is 12.
        seed (Optional[int]): seed value for the random number generator.

        Returns:
        list: list of coordinates with added new points.
        """
        if prev_cords is None:
            prev_cords = []
        
        if seed is not None:
            np.random.seed(seed)

        new_cords = prev_cords.copy()

        std_dev = self.box_y / s_d_koef
        points_new = []

        for _ in range(N_new_points):
            x = np.random.normal(x_s, std_dev)
            y = np.random.normal(y_s, std_dev)

            if 0 < x < self.box_x and 0 < y < self.box_y:
                points_new.append([int(x), int(y)])

        new_cords.extend(points_new)
        self.coordinates = new_cords
        return self.coordinates


    def show_coordinates(self) -> None:
        """
        Визуализация координат точек.
        """
        img_points_koord = np.zeros((self.box_y, self.box_x), dtype=np.uint8)
        for x, y in self.coordinates:
            cv2.circle(img_points_koord, (x, y), radius=4, color=(255, 255, 255), thickness=-1)

        plt.imshow(img_points_koord, cmap='gray')
        plt.axis('off')
        plt.show()