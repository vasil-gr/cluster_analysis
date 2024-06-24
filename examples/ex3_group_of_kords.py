# Пример использования библиотеки для работы с группой координат вокруг заданной точки
from cluster_analysis import CoordinatesGenerator, ClusterAnalyzer

# Инициализация класса CoordinatesGenerator с размером области 2000x1400
K = CoordinatesGenerator(2000, 1400)
# Генерация группы координат вокруг точки (1600, 900)
group_of_kords = K.add_group_of_kords(50, 1600, 900)

# Инициализация класса ClusterAnalyzer с полученными координатами группы
M_group_of_kords = ClusterAnalyzer(group_of_kords, 2000, 1400, r_cut=1)
# Визуализация координат
M_group_of_kords.func_plot_kords()
# Генерация и отображение матрицы кластеров
cluster_matrix_group_of_kords = M_group_of_kords.generate_cluster_matrix(show_map=True)
# Получение списка размеров кластеров
cluster_size_list_group_of_kords = M_group_of_kords.generate_cluster_size_list()
# Генерация и отображение матрицы плотности (без сглаживания)
density_matrix_sm0_group_of_kords = M_group_of_kords.generate_density_matrix(show_density_map=True)
# Вычисление энтропии
entropy = M_group_of_kords.calculate_entropy()
print(f"Entropy: {entropy}")