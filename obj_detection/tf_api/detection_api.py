import os
import tarfile
from os import path
from os.path import realpath, dirname

import numpy as np
import six.moves.urllib as urllib

from Utils import Pipe
from obj_detection.detection import Inference
from obj_detection.tf_api.object_detection.utils import label_map_util
from tf_session.session_runner import SessionRunnable

PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28 = 'faster_rcnn_inception_v2_coco_2018_01_28'
PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17 = 'ssd_mobilenet_v1_coco_2017_11_17'


class TFObjectDetectionAPI(SessionRunnable):

    @staticmethod
    def __get_dir_path():
        return dirname(realpath(__file__))

    @staticmethod
    def __download_model(model_path, download_base, model_file):
        print("downloading model...")
        try:
            os.mkdir(model_path)
        except:
            pass

        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_path + model_file)
        print("finished downloading. extracting...")
        tar_file = tarfile.open(model_path + model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, model_path)
        print("finished extracting.")

    @staticmethod
    def __fetch_model_path(model_name):
        dir_path = TFObjectDetectionAPI.__get_dir_path()
        model_path = dir_path + '/object_detection/pretrained/'
        model_file = model_name + '.tar.gz'
        download_base = 'http://download.tensorflow.org/models/object_detection/'
        path_to_frozen_graph = model_path + model_name + '/frozen_inference_graph.pb'
        if not path.exists(path_to_frozen_graph):
            TFObjectDetectionAPI.__download_model(model_path, download_base, model_file)
        return path_to_frozen_graph

    @staticmethod
    def __fetch_category_indices():
        dir_path = TFObjectDetectionAPI.__get_dir_path()
        path_to_labels = os.path.join(dir_path + '/object_detection/data', 'mscoco_label_map.pbtxt')
        class_count = 90
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                         max_num_classes=class_count,
                                                                         use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

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

    def __init__(self, model_name=PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, graph_prefix=None):

        self.__category_index = self.__fetch_category_indices()
        self.__path_to_frozen_graph = self.__fetch_model_path(model_name)

        if not graph_prefix:
            self.graph_prefix = ''
            graph_prefix = ''
        else:
            self.graph_prefix = graph_prefix + '/'

        self.__in_pipe = Pipe()
        self.__out_pipe = Pipe()

        super(TFObjectDetectionAPI, self).__init__(graph_prefix, self.__path_to_frozen_graph)

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def run(self, tf_sess, tf_default_graph):
        ret, image_np = self.__in_pipe.pull()
        if not ret:
            return

        if self.__in_pipe.is_closed():
            self.__out_pipe.close()
            return

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = tf_default_graph.get_tensor_by_name(self.graph_prefix + 'image_tensor:0')
        boxes = tf_default_graph.get_tensor_by_name(self.graph_prefix + 'detection_boxes:0')
        scores = tf_default_graph.get_tensor_by_name(self.graph_prefix + 'detection_scores:0')
        classes = tf_default_graph.get_tensor_by_name(self.graph_prefix + 'detection_classes:0')
        num_detections = tf_default_graph.get_tensor_by_name(self.graph_prefix + 'num_detections:0')

        (boxes, scores, classes, num_detections) = tf_sess.run([boxes, scores, classes, num_detections],
                                                               feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        num_detections = np.squeeze(num_detections)

        self.__out_pipe.push(Inference(image_np, boxes, scores, classes.astype(np.int32), num_detections, self.__category_index))


