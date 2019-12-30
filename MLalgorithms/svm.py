import numpy as np
from scipy.spatial import distance
np.random.seed(17)


class SVM:
    """Класс реализации алгоритма SVM"""

    def __init__(self, learning_rate=0.001, reg=0.05, epsilon=0.001, max_iter=2000):
        """
        Входные параметры
        __________________
        learning_rate - скорость обучения (0.001 по дефолту)
        reg - регуляризация (0.05 по дефолту)
        max_iter - максимальное количество итераций (2000 по дефолту)
        epsilon - параметр сходимости (0.001 по дефолту)
        """
        self.learning_rate = learning_rate
        self.reg = reg
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.iteration = None
        self.weight_evolution = None

    def fit(self, X, y):
        """
        Входные параметры
        __________________
        X - данные для обучения алгоритма
        y - метки для обучения
        """
        # переходим к меткам {-1,1}
        y_new = y * 2 - 1
        samples, features = X.shape
        # Инициализируем веса, байес, счётчик итераций и начальное значение расстояния между соседними занчениями весов
        self.w = np.random.random(features)
        self.b = 0
        self.iteration = 1
        self.weight_evolution = 2000

        # Обучаем алгоритм пока истинны оба условия
        while self.max_iter >= self.iteration and self.weight_evolution >= self.epsilon:

            w_old = self.w.copy()
            for i, sample in enumerate(X):
                condition = y_new[i] * (np.dot(sample, self.w) - self.b)

                if condition >= 1:
                    self.w -= self.learning_rate * (2 * self.reg * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.reg * self.w - sample * y_new[i])
                    self.b -= self.learning_rate * y_new[i]

            self.weight_evolution = distance.euclidean(w_old, self.w)

            self.iteration += 1

    def predict(self, X):
        """
        Функция предсказания итоговых меток в формате {0, 1}

        Входные параметры:
        __________________
        X - тестовые данные
        """
        return (np.sign(np.dot(X, self.w.T) - self.b) + 1) / 2
