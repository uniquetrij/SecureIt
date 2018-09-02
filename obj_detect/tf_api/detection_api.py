import cv2
from os.path import realpath, dirname
import numpy as np
import os
from os import path
import six.moves.urllib as urllib
import tarfile
import matplotlib
from Utils import Pipe
from obj_detect.tf_api.object_detection.utils import visualization_utils as vis_util

from tf_session.session_runner import SessionRunnable

PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28 = 'faster_rcnn_inception_v2_coco_2018_01_28'
PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17 = 'ssd_mobilenet_v1_coco_2017_11_17'

matplotlib.use('Agg')  # pylint: disable=multiple-statements

# sys.path.append('tf_api/object_detection')

from obj_detect.tf_api.object_detection.utils import label_map_util


class TFObjectDetectionAPI(SessionRunnable):
    __out_pipe = None
    __in_pipe = None

    __object_types = {
        0: None,
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush',
    }

    def __init__(self, model_name=PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, name=''):
        self.name = name
        self.dir_path = dirname(realpath(__file__))
        self.model_path = self.dir_path + '/object_detection/pretrained/'
        self.model_file = model_name + '.tar.gz'
        self.download_base = 'http://download.tensorflow.org/models/object_detection/'
        self.path_to_frozen_graph = self.model_path + model_name + '/frozen_inference_graph.pb'
        path_to_labels = os.path.join(self.dir_path + '/object_detection/data', 'mscoco_label_map.pbtxt')
        self.class_count = 90
        if not path.exists(self.path_to_frozen_graph):
            self.__download()

        self.label_map = label_map_util.load_labelmap(path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.class_count,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.__in_pipe = Pipe()
        self.__out_pipe = Pipe()

        super(TFObjectDetectionAPI, self).__init__(name, self.path_to_frozen_graph)

    def getInPipe(self):
        return self.__in_pipe

    def getOutPipe(self):
        return self.__out_pipe

    def __download(self):
        try:
            os.mkdir(self.model_path)
        except:
            pass
        opener = urllib.request.URLopener()
        opener.retrieve(self.download_base + self.model_file, self.model_path + self.model_file)
        tar_file = tarfile.open(self.model_path + self.model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, self.model_path)

    def run(self, tf_sess, tf_default_graph):
        ret, image_np = self.__in_pipe.pull()
        if not ret:
            return

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = tf_default_graph.get_tensor_by_name(self.name + 'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = tf_default_graph.get_tensor_by_name(self.name + 'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = tf_default_graph.get_tensor_by_name(self.name + 'detection_scores:0')
        classes = tf_default_graph.get_tensor_by_name(self.name + 'detection_classes:0')
        num_detections = tf_default_graph.get_tensor_by_name(self.name + 'num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = tf_sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        print(num_detections)

        annotated = image_np.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            annotated,
            boxes,
            classes.astype(np.int32),
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        cv2.imshow(self.name, annotated)
        cv2.waitKey(1)