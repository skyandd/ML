import numpy as np
from scipy.spatial import distance

np.random.seed(17)


def mean_squared_error(y, y_hat):
    """
    Функция рачёта MSE

    y - метки объектов
    y_hat - предсказанные метки
    """
    return np.mean((y.reshape(1, -1)[0] - y_hat) ** 2)


class LinearRegression:
    """Класс линейной регрессии"""

    def __init__(self, reg=0, lr=0.01, max_iter=300, epsilon=0.001):
        """
        reg - регуляризация
        lr - скорость обучения (learning rate)
        max_iter - максимальное количество итераций
        epsilon - параметр останова
        """

        self.reg = reg
        self.lr = lr
        self.max_iter = max_iter
        self.epsilon = epsilon

    def fit(self, X, y):
        """
        Обучает модель и с помощью градиентного спуска и возвращает историю изменений в history

        X - матрица признаков (N, F), N - количество примеров, F - количество признаков
        y - метки объектов (N,)
        """
        # Добавляем bias
        X = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        # Инициализируем веса
        w = np.random.random(X.shape[1]).reshape(-1, 1)
        # Делаем из меток вектор столбец
        y = y.reshape(-1, 1)
        # Счётчик итераций
        iteration = 1
        # Список в который сохраняется история
        history = []
        # Инициализируем начальное weight_evolution
        weight_evolution = 1000

        # Обучаем пока не будет достигнута максимальное количество итераций или пока не достигним параметра останова
        while iteration <= self.max_iter and weight_evolution > self.epsilon:
            # Вычисляем градиент и предсказанные значения - y_hat,
            # которые будем использовать при расчёте текущего МSE
            y_hat, grad = self._gradient(X, y, w, self.reg)
            # Обновляем weight_evolution
            weight_evolution = distance.euclidean(w, (w - self.lr * grad))
            # Обновляем веса
            w = w - self.lr * grad
            # Обновляем счётчик итераций
            iteration += 1
            # Добавляем MSE в историю
            history.append(mean_squared_error(y, y_hat.reshape(X.shape[0], )))

        self.w = w

        return history

    def predict(self, X_valid):
        """
        Функция predict, выводит предсказаные значения в списке

        X_valid - тестовая матрица признаков
        """
        samples = X_valid.shape[0]
        # Добавляем bias
        X_valid = np.hstack([np.ones(samples).reshape(-1, 1), X_valid])
        # Для предсказания умножаем матрицу X_valid признав на обученные веса
        y_pred = X_valid.dot(self.w)
        return y_pred.reshape(samples, )

    def _gradient(self, X, y, w, reg):
        """
        Функция расчёта градиента, возвращает предсказанные в ходе тренировки метки (y_hat) и
        градиент функции потерь

        Принимает тренировочные данные - X
        Тренировочные метки - y
        Текущие веса при призака - w
        Параметр регуляризации - reg

        """

        # количество обучающих примеров в выборке
        n = X.shape[0]
        # считаем прогноз
        y_hat = X.dot(w)
        # вычисляем ошибку прогноза
        error = y.reshape(-1, 1) - y_hat
        # регуляризованна ошибка
        # X_T_error
        X_T_error = X.T.dot(error)
        # рег grad
        grad = X_T_error * (-1) / n + 2 * reg * w
        # дальше pointwise перемножение - умножаем каждую из координат на ошибку
        return y_hat, grad
