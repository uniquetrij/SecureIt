class Person(object):
    def __init__(self):
        self.__age_list = []
        self.__gender_confidence = {"male": 0, "female": 0}
        self.__gender_list = []

    def detect_age(self):
        return self.__detect_age

    def add_age(self, age):
        self.__age_list.append(age)

    def add_gender(self, gender, confidence):
        self.__gender_list.append(gender)
        if confidence > 0.5:
            self.__gender_confidence['male'] += confidence
        else:
            self.__gender_confidence["female"] += (1 - confidence)

    def get_gender_on_confidence_score(self):
        if self.__gender_confidence['male'] == 0 and self.__gender_confidence['female'] == 0:
            return None
        return 'M' if self.__gender_confidence['male'] > self.__gender_confidence['female'] else 'F'

    def get_age_list(self):
        return self.__age_list

    def get_gender_list(self):
        return self.__gender_list