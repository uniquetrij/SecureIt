import cv2
import numpy as np
from keras.applications import resnet50

from keras import backend as K

class ImageEncoder(object):

    def __init__(self, model, session_runner):
        K.set_session(session_runner.get_session())
        self.__tf_sess = K.get_session()
        self.__model = model
        self.__feature_dim = (2048)
        self.__image_shape = (224, 224, 3)

    def extract_features(self, image, boxes):
        out = np.zeros((len(boxes), self.__feature_dim), np.float32)
        counter = 0
        for box in boxes:
            patch = self.extract_image_patch(image, box, self.__image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., self.__image_shape).astype(np.uint8)
            # cv2.imshow("", patch)
            # cv2.waitKey(0)
            # with K.get_session().as_default():
            img = np.expand_dims(patch, axis=0)
            img = resnet50.preprocess_input(img)
            out[counter] = self.__model.observe(img)


            counter+=1
        return out

    def extract_image_patch(self, image, bbox, patch_shape):
        sx, sy, ex, ey = np.array(bbox).astype(np.int)
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))

        return image