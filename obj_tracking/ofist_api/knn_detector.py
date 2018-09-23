import numpy as np
from collections import Counter


class DistanceMetric:
    @staticmethod
    def cosine_distance(x, y):
        return 1 - np.dot(x, y.T) / (np.sqrt(np.dot(x, x.T)) * np.sqrt(np.dot(y, y.T)))

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

class KnnDetector:
    def __init__(self):
        self.__X = []
        self.__Y = []
        self.__D = None
        self.__Z = None

    def update(self, X, Y):
        for i in range(len(X)):
            self.__X.append(X[i])
            self.__Y.append(Y[i])
        self.__D = None
        self.__Z = None
        return self

    def observe(self, f_vec, distance_metric=DistanceMetric.cosine_distance):
        self.__D = []
        for x in self.__X:
            self.__D.append(distance_metric(x, f_vec))
        self.__D, self.__Z = KnnDetector.sort_list(self.__D, self.__Y)
        return self

    def get_nearest(self, k=None):
        return self.__Z[:k], self.__D[:k]

    @staticmethod
    def sort_list(sort_by_values, list_to_sort):
        return zip(*sorted(zip(sort_by_values, list_to_sort)))


if __name__ == '__main__':
    obj = KnnDetector()
    X = np.array([[6.6, 6.2, 1],
                  [9.7, 9.9, 2],
                  [8.0, 8.3, 2],
                  [6.3, 5.4, 1],
                  [1.3, 2.7, 0],
                  [2.3, 3.1, 0],
                  [6.6, 6.0, 1],
                  [6.5, 6.4, 1],
                  [6.3, 5.8, 1],
                  [9.5, 9.9, 2],
                  [8.9, 8.9, 2],
                  [8.7, 9.5, 2],
                  [2.5, 3.8, 0],
                  [2.0, 3.1, 0],
                  [1.3, 1.3, 0]])

    Y = X[:, 2]
    X = X[:, :2]
    obj.update(X, Y)
    obj.observe(X[12], distance_metric=DistanceMetric.cosine_distance)
    print(obj.get_nearest(7))
    obj.observe(X[12], distance_metric=DistanceMetric.euclidean_distance)
    print(obj.get_nearest(7))
