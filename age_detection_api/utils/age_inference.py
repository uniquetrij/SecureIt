import cv2
import numpy as np

class AgeInference(object):

    '''
    AgeInference class is to store the results of all faces present in a frame after processing it
    through the age_api model.
    '''

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

    def get_annotated(self,):
        '''
        :return:Processed and annotated Image for display
        '''
        # temp = None
        for i, bbox in enumerate(self.__bboxes):
            self.__annotated_image = self.annotate(self.__annotated_image, bbox, self.__ages[i], self.__genders[i], self.__ethnicity[i])
        return self.__annotated_image

    def get_crop(self, bbox):
        return self.__image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]),:]

    def annotate(self, image, bbox, age, gender, ethnicity):
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.__bbox_color, 2)
        # label = "{},{},{}".format(int(age),
        #                           "F" if gender[0] > 0.5 else "M",
        #                           AgeInference.ETHNICITY[ethnicity])

        label = "{},{}".format(int(age),
                                  "F" if gender[0] > 0.5 else "M")

        image = AgeInference.draw_label(image, (int(bbox[0]), int(bbox[1])), label)
        return image

    def set_bboxes(self, bboxes):
        self.__bboxes = bboxes

    def set_ages(self, ages):
        self.__ages = ages

    def set_genders(self, genders):
        self.__genders = genders

    def set_ethnicity(self, ethnicity):
        self.__ethnicity = ethnicity

    @staticmethod
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        return image

