import cv2
import numpy as np

class AgeInference(object):
    ETHNICITY = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
    def __init__(self, image, bboxes = None, ages = None, genders = None, ethnicity = None):
        self.__image = image
        self.__bboxes = bboxes
        self.__ages = ages
        self.__genders = genders
        self.__ethnicity = ethnicity
        self.__annotated_image = np.copy(self.__image)
        self.__text_color = (255,0,0)
        self.__bbox_color = (255,255,0)
    def get_image(self):
        return self.__image

    def get_bboxes(self):
        return self.__bboxes

    def get_ages(self):
        return self.__ages

    def get_genders(self):
        return self.__genders

    def get_ethnicity(self):
        return self.__ethnicity

    def get_annotated(self):
        for i, bbox in enumerate(self.__bboxes):
            cv2.rectangle(self.__annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.__bbox_color, 2)
            label = "{},{},{}".format(int(self.__ages[i]),
                                      "F" if self.__genders[i][0] > 0.5 else "M", AgeInference.ETHNICITY[self.__ethnicity[i]])
            self.draw_label( (self.__bboxes[i][0], self.__bboxes[i][1]), label)

        return self.__annotated_image

    def set_bboxes(self, bboxes):
        self.__bboxes = bboxes

    def set_ages(self, ages):
        self.__ages = ages

    def set_genders(self, genders):
        self.__genders = genders

    def set_ethnicity(self, ethnicity):
        self.__ethnicity = ethnicity


    def draw_label(self,point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(self.__annotated_image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(self.__annotated_image, label, point, font, font_scale, (255, 255, 255), thickness)


