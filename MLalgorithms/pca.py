import numpy as np


class PCA:
    """Релаизация метода главных компонент"""

    def __init__(self, n_components):
        """
        Входные параметры:
        _______________________
        n_components - количество главныех компонент
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        """
        Обучение алгорита

        Входные параметры:
        _____________________
        X - данные для обучения алгоритма
        """

        # Находим среднее, для центрирования данных
        self.mean = np.mean(X, axis=0)
        # Центрируем
        X_scale = X - self.mean
        # Находим матрицу ковариации
        cov = np.cov(X_scale.T)
        # Находим собственные вектора и значения
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        # собственные значение, конечно уже отсортированы по убыванию, но на всяких, сделам 100% вариант
        idx = np.argsort(eigenvalues)[::-1]
        # проиндексируем данные по возрастанию
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        # и выберем нужное количество клавных компонент
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        """
        Функция трансформировния данных

        Входные параметры:
        _________________
        X - данные для трансформации
        """
        # масштабируем данные
        X_scale = X - self.mean
        # Возвращаем проекции на наши главные компоненты
        return np.dot(X_scale, self.components.T)
