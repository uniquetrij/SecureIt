import numpy as np
from collections import Counter

class KNNCosinePredictor:
    def __init__(self):
        self.__class = None
        self.__similarity = None

    def fit(self, X, Y):
        self.__X = X
        self.__Y = Y
        self.__nearest_classes = None
        self.__nearest_distances = None
        self.__bin_count = None

    def predict(self, f_vec, nearest_count=5):
        self.__nearest_distances = []
        for x in self.__X:
            self.__nearest_distances.append(self.get_cosine_distance(x, f_vec))

        Z, Y = KNNCosinePredictor.sort_list(self.__Y.copy(), self.__nearest_distances)

        self.__nearest_classes = np.array(Y[:nearest_count])
        self.__nearest_distances = np.array(Z[:nearest_count])

    def get_best_distance(self):
        indices  = np.where(self.__nearest_classes == self.get_best_class())
        avg = 0
        for i in indices[0]:
            avg+=self.__nearest_distances[i]
        return  avg/len(indices[0])

    def get_nearest_classes(self):
        return self.__nearest_classes

    def get_nearest_distances(self):
        return self.__nearest_distances

    def get_best_class(self):
        return Counter(self.__nearest_classes).most_common(1)[0][0]

    @staticmethod
    def get_cosine_distance(a, b):
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        dot = 1-np.dot(a, b.T)[0][0]
        return dot

    @staticmethod
    def sort_list(list_to_sort, list_sort_by):
        zipped_pairs = zip(list_sort_by, list_to_sort)
        Z, Y = zip(*sorted(zipped_pairs))
        return Z, Y


if __name__ == '__main__':
    obj = KNNCosinePredictor()
    obj.fit([[1, 0], [0, 1], [-1, 0], [0, -1]],[1, 0, 0, 3])
    obj.predict([2, 3], 3)
    print(obj.get_nearest_classes())
    print(obj.get_nearest_distances())
    print(obj.get_best_class())
    print(obj.get_best_distance())




    # print(obj.get_best_similarity())

    # print(obj.get_nearest_similarity())
    # print(obj.get_nearest_class())