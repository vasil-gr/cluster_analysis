# Пример использования библиотеки для работы с идеально распределёнными координатами
from cluster_analysis import CoordinatesGenerator, ClusterAnalyzer

# Инициализация класса CoordinatesGenerator с размером области 2000x1400
K = CoordinatesGenerator(2000, 1400)
# Генерация идеально распределённых координат (12 точек по оси X)
ideal_kords = K.generate_ideal_kords(12)

# Инициализация класса ClusterAnalyzer с полученными идеальными координатами
M_ideal_kords = ClusterAnalyzer(ideal_kords, 2000, 1400, r_cut=1)
# Визуализация координат
M_ideal_kords.func_plot_kords()
# Генерация и отображение матрицы кластеров
cluster_matrix_ideal_kords = M_ideal_kords.generate_cluster_matrix(show_map=True)
# Получение списка размеров кластеров
cluster_size_list_ideal_kords = M_ideal_kords.generate_cluster_size_list()
# Генерация и отображение матрицы плотности (без сглаживания)
density_matrix_sm0_ideal_kords = M_ideal_kords.generate_density_matrix(show_density_map=True)
# Генерация и отображение матрицы плотности (со сглаживанием)
# density_matrix_sm10_ideal_kords = M_ideal_kords.generate_density_matrix(smo_num=10, show_density_map=True)
# Вычисление энтропии
entropy = M_ideal_kords.calculate_entropy()
print(f"Entropy: {entropy}")