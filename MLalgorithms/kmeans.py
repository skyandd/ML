import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)


def _create_cluster_index(sample, centroids):
    """Функция рачёта расстояний от сэмпла до каждого кластера. Выводит индекс того, чьё растояние минимально"""

    return np.argmin([euclidean_distance(sample, x) for x in centroids])


class KMeans:
    """Класс алгоритма кластеризации K-means"""

    def __init__(self, K, max_iter=200, plot_steps=False):
        """Воходные параметры:
         K - количество кластеров в данных
         max_iter - максимальное количество итераций до сходимости. По дефолту 200
         plot_steps - флажок, строить графики во время обучения или нет
         """

        self.K = K
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        self.centroids_list = [[] for i in range(self.K)]

    def fit(self, X):
        """ Обучение модели. По завершению сохраняет обученные координаты центроидов в память

        Входные параметры:
        X - тренировочные данные
        """
        self.samples, self.features = X.shape
        self.X = X
        # инициализируем кластера случайными сэмпламы из набора данных
        centroids_idx = np.random.choice(self.samples, self.K, replace=False)
        self.centroids = [X[i] for i in centroids_idx]

        # Обучаем модель в цикле.
        for _ in range(self.max_iter):

            # Сохраняем список из K-cписков кластеров
            self.clusters = self._create_clusters(self.centroids)

            # строим график, если установлен флажок
            if self.plot_steps:
                self.plot()

            # Сохраняем старые координаты центроидов
            centroids_old = self.centroids

            # Переназначаем переменную другими координами
            self.centroids = self._center_centroids(self.clusters)

            # Если изменение координат не происходит, то заканчиваем обучение
            if self._stop_function(self.centroids, centroids_old):
                break

    def _create_clusters(self, centroids):
        """Функция определния список из K-списков с индексами сэмплов отнесенных к определенным кластерам"""

        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = _create_cluster_index(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _center_centroids(self, clusters):
        """Функция рачёта координат центров новых центроидов"""

        return [self.X[cluster].mean(axis=0) for cluster in clusters]

    def _stop_function(self, centroids, centroids_old):
        """Функция расчёта дельты смещения координат кластеров"""
        distance = [euclidean_distance(centroids[i], centroids_old[i]) for i in range(self.K)]
        return sum(distance) == 0

    def _in_label(self, clusters):
        # Функция перевода переменной clusters в метки
        label = np.empty(self.X.shape[0])

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                label[sample_idx] = cluster_idx
        return label

    def predict(self, X):
        """Предсказывает метки для входных данные

        Входные параметры:
        X - тестовые сэмплы
        """
        self.X = X
        clusters = self._create_clusters(self.centroids)
        return self._in_label(clusters)

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()