import cv2

import numpy as np


class ObjectDetectorInterface:
    def infer(self, image, types=None):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def getInPipe(self):
        pass

    def getOutPipe(self):
        pass

class InstanceType:
    __type = None
    __detectorId = None

    def __init__(self, type, detectorId=None):
        self.__type = type
        self.__detectorId = detectorId

    def getType(self):
        return self.__type

    def getDetectorId(self):
        return self.__detectorId


class InferenceBounds:
    __y_tl = None
    __x_tl = None
    __y_br = None
    __x_br = None

    def __init__(self, x_tl, y_tl, x_br, y_br):
        self.__y_tl, self.__x_tl, self.__y_br, self.__x_br = y_tl, x_tl, y_br, x_br

    def getTop(self):
        return self.__y_tl

    def getBottom(self):
        return self.__y_br

    def getLeft(self):
        return self.__x_tl

    def getRight(self):
        return self.__x_br

    def getBoundingBox(self):
        return [[self.__x_tl, self.__y_tl],
                [self.__x_tl, self.__y_br],
                [self.__x_br, self.__y_br],
                [self.__x_br, self.__y_tl]]

    def getBoundingCoordinates(self):
        return self.__y_tl , self.__x_tl, self.__y_br, self.__x_br, self.__y_br




class Inference:
    __image = None
    __decisionInstances = None
    __annotatedImage = None
    __crowdCount = None

    def __init__(self, image, decisionInstances, annotatedImage, crowdCount):
        self.__image = image
        self.__decisionInstances= decisionInstances
        self.__annotatedImage = annotatedImage
        self.__crowdCount = crowdCount

    def getAnnotatedImage(self):
        return self.__annotatedImage

    def getImage(self):
        return self.__image.copy()

    def getDecisionInstances(self):
        return self.__decisionInstances

    def getCrowdCount(self):
        return self.__crowdCount

class DecisionInstance:
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
    __inferences = None

    def __to_np_array(self, image):
        im_width, im_height, _ = image.shape
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def __init__(self, image, detector):
        self.__image = image.copy()
        self.__detector = detector
        self.__inferences = detector.infer(self.__image.copy())

    def getInference(self, index):
        return self.__inferences[index]

    def length(self):
        return len(self.__inferences)

    def getInstanceImage(self, index, types=None, threshold = None):
        if threshold is None:
            threshold = 0.3
        inference = self.getInference(index)
        if types is None or (inference.getClass().getType() in types and inference.getScore() > threshold):
            bbox = inference.getBox()
            return self.__image[int(bbox.getTop()):int(bbox.getBottom()), int(bbox.getLeft()):int(bbox.getRight())]
        else:
            return None

    def getAnnotatedImage(self, types=None, threshold=None):
        if threshold is None:
            threshold = 0.3
        img = self.__image.copy()
        for i in range(self.length()):
            inference = self.getInference(i)
            if inference.getClass().getType()in types and inference.getScore() > threshold:
                bbox = inference.getBox()
                img = cv2.polylines(img, [np.int32(bbox.getBoundingBox())], 1, (0, 255, 0), 3)
        return img














