import cv2
from keras.applications import resnet50


from keras.preprocessing import image
from keras.applications.vgg16 import vgg16
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
import numpy as np

from scipy.spatial import distance
from keras.models import Model
from keras.applications import resnet50
import cv2
from random import shuffle


import numpy as np

class ImageEncoder(object):

    def __init__(self):
        self.model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        # print("Model Loaded")
        self.feature_dim = (2048)
        self.image_shape = (224,224, 3)

    def extract_features(self, image, boxes):

        out = np.zeros((len(boxes), self.feature_dim), np.float32)
        counter = 0
        for box in boxes:
            patch = self.extract_image_patch(image, box, self.image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., self.image_shape).astype(np.uint8)
            # cv2.imshow("", patch)
            # cv2.waitKey(0)
            img = np.expand_dims(patch, axis=0)
            img = resnet50.preprocess_input(img)

            out[counter] = self.model.observe(img)
            counter+=1
        return out

    def extract_image_patch(self, image, bbox, patch_shape):
        """Extract image patch from bounding box.

        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped  # predict(img)  # for kal!
                # print(pos)at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.

        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.

        """
        # bbox = np.array(bbox)
        # if patch_shape is not None:
        #     # correct aspect ratio to patch shape
        #     target_aspect = float(patch_shape[1]) / patch_shape[0]
        #     new_width = target_aspect * bbox[3]
        #     bbox[0] -= (new_width - bbox[2]) / 2
        #     bbox[2] = new_width
        #
        # # convert to top left, bottom right
        # bbox[2:] += bbox[:2]
        # bbox = bbox.astype(np.int)
        #
        # # clip at image boundaries
        # bbox[:2] = np.maximum(0, bbox[:2])
        # bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        # if np.any(bbox[:2] >= bbox[2:]):
        #     return None
        sx, sy, ex, ey = np.array(bbox).astype(np.int)
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))

        return image