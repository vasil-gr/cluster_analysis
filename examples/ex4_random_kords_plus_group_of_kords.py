# Пример использования библиотеки для работы со случайными координатами и группой координат
from cluster_analysis import Kords, Main_1

# Инициализация класса Kords с размером области 2000x1400
K = Kords(2000, 1400)
# Генерация 100 случайных координат
random_kords_100 = K.generate_random_kords(100)

# Добавление группы координат к случайным координатам
random_kords_100_plus_group_of_kords = K.add_group_of_kords(50, 1600, 900, random_kords_100)
# Инициализация класса Main_1 с полученными координатами
M_random_kords_plus_group_of_kords = Main_1(random_kords_100_plus_group_of_kords, 2000, 1400, r_cut=1)
# Визуализация координат
M_random_kords_plus_group_of_kords.func_plot_kords()
# Генерация и отображение матрицы кластеров
cluster_matrix_random_kords_plus_group_of_kords = M_random_kords_plus_group_of_kords.generate_cluster_matrix(show_map=True)
# Получение списка размеров кластеров
cluster_size_list_random_kords_plus_group_of_kords = M_random_kords_plus_group_of_kords.generate_cluster_size_list()
# Генерация и отображение матрицы плотности (без сглаживания)
density_matrix_sm0_random_kords_plus_group_of_kords = M_random_kords_plus_group_of_kords.generate_density_matrix(show_density_map=True)
# Вычисление энтропии
entropy = M_random_kords_plus_group_of_kords.calculate_entropy()
print(f"Entropy: {entropy}")