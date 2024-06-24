# Пример использования библиотеки для работы со случайными координатами
from cluster_analysis import CoordinatesGenerator, ClusterAnalyzer

# Инициализация класса CoordinatesGenerator с размером области 2000x1400
K = CoordinatesGenerator(2000, 1400)
# Генерация 100 случайных координат
random_kords_100 = K.generate_random_kords(100)

# Инициализация класса ClusterAnalyzer с полученными случайными координатами
M_random_kords = ClusterAnalyzer(random_kords_100, 2000, 1400, r_cut=1)
# Визуализация координат
M_random_kords.func_plot_kords()
# Генерация и отображение матрицы кластеров
cluster_matrix_random_kords = M_random_kords.generate_cluster_matrix(show_map=True)
# Получение списка размеров кластеров
cluster_size_list_random_kords = M_random_kords.generate_cluster_size_list()
# Генерация и отображение матрицы плотности (без сглаживания)
density_matrix_sm0_random_kords = M_random_kords.generate_density_matrix(show_density_map=True)
# Генерация и отображение матрицы плотности (со сглаживанием)
# density_matrix_sm10_random_kords = M_random_kords.generate_density_matrix(smo_num=10, show_density_map=True)
# Вычисление энтропии
entropy = M_random_kords.calculate_entropy()
print(f"Entropy: {entropy}")