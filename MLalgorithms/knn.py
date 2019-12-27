import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean

np.random.seed(17)


class KNNClassifier:
    """Реализация алгоритма k ближайших соседей"""

    def __init__(self, n_neighbors=2):
        """
        Простая реализация KNNClassifier

        Параметры
        ----------
        n_neighbors: количество ближайших соседей (2 по дефолту)
        distance_func: функция, euclidean - по дефолту, принимает на вход 2 аргумента
        """
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        """
        фукнция fit, запоминает входные данные для дальнейшего обучения
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Функция predict, предсказывает метки для X_test
        """
        # Предсказывает ближайших соседей для каждого объекта из X_test
        predictions = [self.predict_x(x) for x in X_test]
        # Считается количество уникальных элементов и выводится метка самоего часто встречающегося
        label = np.array([max(Counter(i)) for i in predictions])
        return label

    def predict_x(self, x):
        """
        Функция predict_x, возвращает метки k-соседей объекта
        """
        # Считает расстояние от тестового объекта до каждого тренировочного,
        # затем каждому расстоянию присваиваетя метка.
        # После чего список [(расстояние, метка)...] сортируется по возрастанию
        distance = sorted(zip([euclidean(x, example) for example in self.X_train], self.y_train),
                          key=lambda x: x[0])
        # C отсортированного списка выбирается n_neighbors первых элеметнтов
        neighbors_target = [target for (_, target) in distance[:self.n_neighbors]]
        return neighbors_target
