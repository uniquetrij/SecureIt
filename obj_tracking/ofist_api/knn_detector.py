from operator import itemgetter

import numpy as np
from collections import defaultdict


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

    def obtain_old(self, k=5, n=None):
        Z, D = self.__Z[:k], self.__D[:k]
        dict = {}
        for i in range(len(Z)):
            if Z[i] in dict.keys():
                dict[Z[i]][0] += 1
                dict[Z[i]][1] = min(D[i],dict[Z[i]][1])
            else:
                dict[Z[i]] = [1, D[i]]

        # for i in range(len(dict)):
        #     dict[Z[i]][1] /= dict[Z[i]][0]
        if n is not None:

            if n > 0:
                while n > 1:
                    del dict[max(dict.items(), key=itemgetter(1))[0]]
                    n -= 1
                ret = max(dict.items(), key=itemgetter(1))
            if n < 0:
                while n < -1:
                    del dict[min(dict.items(), key=itemgetter(1))[0]]
                    n += 1
                ret = tuple(min(dict.items(), key=itemgetter(1)))
            return (ret[0], ret[1][0], ret[1][1])

        return dict

    def obtain(self, k=5):
        Z, D = self.__Z[:k], self.__D[:k]
        dict = {}
        for i in range(len(Z)):
            if Z[i] in dict.keys():
                dict[Z[i]][0] += 1
                dict[Z[i]][1] = min(D[i], dict[Z[i]][1])
                dict[Z[i]][2] += D[i]
                dict[Z[i]][3] = max(D[i], dict[Z[i]][3])
            else:
                dict[Z[i]] = [1, D[i], D[i], D[i]]

        for key in dict.keys():
            dict[key][2] /= dict[key][0]

        self.__obtain = {0: dict}
        return dict

    def get(self, m):
        dict = self.__obtain.copy()
        if m not in dict.keys():
            n = m
            if n is not None:
                if n > 0:
                    while n > 1:
                        del dict[0][max(dict[0].items(), key=itemgetter(1))[0]]
                        n -= 1
                    dict[m] = max(dict[0].items(), key=itemgetter(1))
                if n < 0:
                    while n < -1:
                        del dict[0][min(dict[0].items(), key=itemgetter(1))[0]]
                        n += 1
                    dict[m] = tuple(min(dict[0].items(), key=itemgetter(1)))
        return dict[m][0], dict[m][1][0], dict[m][1][1], dict[m][1][2], dict[m][1][3]

    @staticmethod
    def sort_list(sort_by_values, list_to_sort):
        return zip(*sorted(zip(sort_by_values, list_to_sort)))

    # @staticmethod
    # def get_best_count(knn):
    #     id, distance = knn
    #     dict = {}
    #     for i in range(len(id)):
    #         if id[i] in dict.keys():
    #             dict[id[i]][0] += 1
    #             dict[id[i]][1] += distance[i]
    #         else:
    #             dict[id[i]] = [1, distance[i]]
    #
    #     for i in range(len(dict)):
    #         print(i, dict[id[i]])
    #         dict[id[i]][2] /= dict[id[i]][0]
    #     return dict


if __name__ == '__main__':
    obj = KnnDetector()
    X = np.array([[6.6, 6.2, "B"],
                  [9.7, 9.9, "C"],
                  [8.0, 8.3, "C"],
                  [6.3, 5.4, "B"],
                  [1.3, 2.7, "D"],
                  [2.3, 3.1, "D"],
                  [6.6, 6.0, "B"],
                  [6.5, 6.4, "B"],
                  [6.3, 5.8, "B"],
                  [9.5, 9.9, "C"],
                  [8.9, 8.9, "C"],
                  [8.7, 9.5, "C"],
                  [2.5, 3.8, "D"],
                  [2.0, 3.1, "D"],
                  [1.3, 1.3, "D"]])

    Y = X[:, 2]
    X = X[:, :2].astype(np.float)
    print(obj.update(X, Y).observe(X[12], distance_metric=DistanceMetric.euclidean_distance).obtain(10))
    print(obj.obtain_old(10))
    # print(obj.get(1))
    # print(obj.get_k_nearest(7))
    # obj.observe(X[12], distance_metric=DistanceMetric.euclidean_distance)
    # print(obj.get_k_nearest(7))
    # print(obj.get_best_count(obj.get_k_nearest(7)))
