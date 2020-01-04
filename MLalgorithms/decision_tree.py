import numpy as np
from collections import Counter

np.random.seed(17)


class Node:
    """Вспомогательный класс узла"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Входные параметры:
        ______________________
        feature - признак размиения на узле
        threshold - порог разбиения на узле
        left - узел слева
        right - узел справа
        value - метка узла
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def check_value(self):
        """Проверка узла на наличие в нём value"""

        return self.value is not None


def _entropy(y):
    """Функия расчёта энтропии"""

    count_i = np.bincount(y)
    prob_i = count_i / len(y)

    return -np.sum([p * np.log2(p) for p in prob_i if p > 0])


class DecisionTree:
    """Класс реализации дерева решений"""

    def __init__(self, max_depth=10, min_sample_leaf=2, max_features=None):
        """
        Входные параметры:
        _______________________
        max_depth - максимальная глубина дерева (10 по дефолту)
        min_sample_leaf - минимальное количество примеров на листе (2 по дефолту)
        max_feature - число признаков для поиска лучшего разбиения (None по дефолту)
        """
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.max_feature = max_features
        self.tree = None

    def fit(self, X, y):
        """Функция обучения алгоритма"""

        # Если макс количество признаков не установлено, то делаем его следующим
        if self.max_feature is None:
            self.max_feature = np.int(round(np.log2(X.shape[1])))
        # Обучаем дерево
        self.tree = self._grow(X, y)

    def predict(self, X):
        """Функция предсказания итоговых меток тестовых данных"""

        return np.array([self._predict_label(sample, self.tree) for sample in X])

    def _grow(self, X, y, depth=0):
        """Вспомогательная рекурсивная функция"""

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # Если выполняется параметр останова или в листе только одна метка, то возвращаем самую популярную
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_sample_leaf):
            lable_leef = max(Counter(y))

            return Node(value=lable_leef)

        # Рандомно выбираем признаки
        feat_idx = np.random.choice(n_features, self.max_feature, replace=False)
        # Находим лучшей признак для разбиение и его порог
        best_feature, best_threshold = self._best_separation(X, y, feat_idx)
        # Получаем индексы для новых узлов
        left_idx, right_idx = self._get_indexes(X[:, best_feature], best_threshold)
        # Передаём в следущие узлы разделенные данные
        left = self._grow(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow(X[right_idx, :], y[right_idx], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_separation(self, X, y, n_feats):
        """Вспомогательная функия для поиска лучшего разбиения"""
        best_ig = -1
        best_feature_idx = None
        best_threshold = None
        # Перебераем все признаки и пороги, ищем лучшие
        for feat in n_feats:

            X_colum = X[:, feat]

            for threshold in X_colum:

                ig = self._information_gain(X_colum, y, threshold)

                if ig > best_ig:
                    best_ig = ig
                    best_feature_idx = feat
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _information_gain(self, X_colum, y, threshold):
        """Вспомогательная функия расёта прироста информации"""
        # Считаем энтропию начального состояни
        parent_entropy = _entropy(y)
        # По заданному порогу делим данные и считаем IG
        left_idx, right_idx = self._get_indexes(X_colum, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        l_share, r_share = len(left_idx) / len(y), len(right_idx) / len(y)
        l_entropy, r_entropy = _entropy(y[left_idx]), _entropy(y[right_idx])

        ig = parent_entropy - l_share * l_entropy - r_share * r_entropy

        return ig

    def _get_indexes(self, X, threshold):
        """Вспомагательная функия для получения индексов левой и правой части по заданному порогу"""

        left_idx = np.argwhere(X <= threshold).flatten()
        right_idx = np.argwhere(X > threshold).flatten()

        return left_idx, right_idx

    def _predict_label(self, sample, tree):
        """Вспомогательная, рекурсивная функция для обхода дерва"""
        if tree.check_value():
            return tree.value
        # Если в узле нет значения, то идём дляьше по дереву с учетом условий
        if sample[tree.feature] <= tree.threshold:
            return self._predict_label(sample, tree.left)
        return self._predict_label(sample, tree.right)
