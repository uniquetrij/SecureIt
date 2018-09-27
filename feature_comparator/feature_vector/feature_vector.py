from collections import OrderedDict


class FeatureVector:
    def __init__(self):
        self.__vector_dict = OrderedDict()

    def add_vector(self, v_id, vector):
        if v_id  not in self.__vector_dict:
            self.__vector_dict[v_id] = []
        self.__vector_dict[v_id].append(vector)

    def get_vector_dict(self):
        return self.__vector_dict

