import numpy as np


class ObjectDetectorInterface:
    def infer(self, image):
        pass


class ObjectInstance:
    __objectClass = None
    __boundingPath = None
    __confidenceScore = None
    __decisionMask = None

    def __init__(self, decisionClass, confidenceScore, boundingPath, decisionMask=None):
        self.__objectClass = decisionClass
        self.__boundingPath = boundingPath
        self.__confidenceScore = confidenceScore
        self.__decisionMask = decisionMask

    def getClass(self):
        return self.__objectClass

    def getBox(self):
        return self.__boundingPath

    def getScore(self):
        return self.__confidenceScore


class ObjectDetector:
    __image = None
    __detector = None
    __objectInstances = None
    __length = None

    def __to_np_array(self, image):
        im_width, im_height, _ = image.shape
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def __init__(self, image, detector):
        self.__image = image
        self.__detector = detector
        self.__objectInstances = detector.infer(self.__image)

    def getInstance(self, index):
        return self.__objectInstances[index]

    def length(self):
        return len(self.__objectInstances)










