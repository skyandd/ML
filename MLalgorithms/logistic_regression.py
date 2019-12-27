import numpy as np
from scipy.spatial import distance

np.random.seed(17)


def sigmoid_fit(X, w, y):
    """
    Вспомгательная функция сигмоиды от -X.Tw*y для fit

    Применяется только в обучении
    """
    return 1 / (1 + np.exp(-1 * (X.dot(w.T) * y)))


def sigmoid_predict(X_test, w):
    """
    Вспопогательная функция сигмоиды от -Xw для predict_proba

    Применяется только во время предсказания меток классов в def predict_proba
    """
    return 1 / (1 + np.exp(-1 * (X_test.dot(w.T))))


def log_los(X, w, y):
    """
    Вспомогательная функция расчёта минимизируемого функционала
    """

    return np.mean(np.log((1 + np.exp(-X.dot(w.T) * y))))


class LogisticRegresson:
    """Реализация логистической регрессии на питон"""

    def __init__(self, reg=1, lr=0.01, max_iter=200, epsilon=0.01):
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

    def fit(self, X_train, y_train):
        """
        Обучает модель и с помощью градиентного спуска и возвращает историю изменений в history

        X_train - матрица признаков (N, F), N - количество примеров, F - количество признаков
        y_train - метки объектов (N,)
        """

        # Преобразуем y_train в вектор столбец размерностью [n:1], где n - количество примеров обучения
        y_train = y_train.reshape(-1, 1)
        # Переходим от меток {0, 1} к {-1,1}
        y_train = 2 * y_train - 1
        # Инициализируем веса
        w = np.random.random(X_train.shape[1]).reshape(1, -1)
        # Заводим счётчик итераций
        iteration = 1
        # Объявляем список с историей изменений loglos
        history = []
        # Инициализируем начальное состояние расстояние между двумя разными состояниями вектора W
        weight_evolution = 2000
        # Находим градиент, weight_evolution, обновляем веса, и добавляем los в список
        # пока количество итераций не достигнет максимума или наш алгоритм не сойдётся
        while iteration <= self.max_iter and weight_evolution > self.epsilon:
            # Находим регуляризованный градиент
            grad = self.gradient(X_train, w, y_train, self.reg)
            # Обновляем weight_evolution
            weight_evolution = distance.euclidean(w, (w - self.lr * grad))
            # Обновляем веса
            w = w - self.lr * grad
            # Добавляем los в историю
            history.append(log_los(X_train, w, y_train))
            # Обновляем счётчик итераций
            iteration += 1

        self.w = w
        self.history = history
        return history

    def gradient(self, X, w, y, reg):
        """
        Функция расчёта градиента los функции

        Принимает тренировочные данные - X
        Тренировочные метки - y
        Текущие веса при призака - w
        Параметр регуляризации - reg
        """

        C = 1 / self.reg
        grad = X.T.dot(y * sigmoid_fit(X, w, y)) + C * 2 * w.T
        return grad.T

    def predict_proba(self, X_test):
        """
        функция predict_proba, выводит вероятность пренадлежности к каждому классу

        Принимает на вход X_test и использует обученные веса
        """

        predict_list = np.zeros((X_test.shape[0], 2))
        predict_list[:, 0] = 1 - sigmoid_predict(X_test, self.w)[:, 0]
        predict_list[:, 1] = sigmoid_predict(X_test, self.w)[:, 0]
        return predict_list

    def predict(self, X_test):
        """
        Функция predict, возвращает предсказаные метки принадлежности к каждому классу

        Принимает на вход X_test и использует обученные веса
        """

        predict = (X_test.dot(self.w.T) < 0).astype('int')
        return predict.reshape(1, -1)[0]
