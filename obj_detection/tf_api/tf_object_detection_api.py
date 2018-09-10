import os
import tarfile
from os import path
from os.path import realpath, dirname
from threading import Thread
from time import sleep

import numpy as np
from obj_detection.tf_api.object_detection.utils import ops as utils_ops
import six.moves.urllib as urllib
import tensorflow as tf

from tf_session.tf_session_utils import Pipe
from obj_detection.obj_detection_utils import Inference
from obj_detection.tf_api.object_detection.utils import label_map_util
from tf_session.tf_session_runner import SessionRunnable

PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28 = 'faster_rcnn_inception_v2_coco_2018_01_28'
PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17 = 'ssd_mobilenet_v1_coco_2017_11_17'
PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28 = 'mask_rcnn_inception_v2_coco_2018_01_28'


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

    __class_labels_dict = {
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
        37: 'sports classesball',
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

    def __init__(self, session_runner, model_name=PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, image_shape=None,
                 graph_prefix=None, flush_pipe_on_read=False):
        self.__tf_sess = session_runner.get_session()
        self.__category_index = self.__fetch_category_indices()
        self.__path_to_frozen_graph = self.__fetch_model_path(model_name)
        self.__flush_pipe_on_read = flush_pipe_on_read
        self.__image_shape = image_shape
        self.__session_runner = session_runner
        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        if not graph_prefix:
            self.graph_prefix = ''
            graph_prefix = ''
        else:
            self.graph_prefix = graph_prefix + '/'

        super(TFObjectDetectionAPI, self).__init__(graph_prefix, self.__path_to_frozen_graph)

        self.init_graph()


    def __in_pipe_process(self, image):
        return image

    def __out_pipe_process(self, inference):
        image_np, output_dict = inference
        num_detections = int(output_dict['num_detections'][0])
        detection_classes = output_dict['detection_classes'][0][:num_detections].astype(np.uint8)
        detection_boxes = output_dict['detection_boxes'][0][:num_detections]
        detection_scores = output_dict['detection_scores'][0][:num_detections]
        if 'detection_masks' in output_dict:
            detection_masks = output_dict['detection_masks'][0][:num_detections]
        else:
            detection_masks = None

        return Inference(image_np, num_detections, detection_boxes, detection_classes, detection_scores,
                         detection_masks, self.__category_index, self.__class_labels_dict)

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def init_graph(self):
        with self.__tf_sess.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.get_path_to_frozen_graph(), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name=self.get_graph_prefix())

        tf_default_graph = self.__tf_sess.graph

        self.__image_tensor = tf_default_graph.get_tensor_by_name(self.graph_prefix + 'image_tensor:0')
        tensor_names = {output.name for op in tf_default_graph.get_operations() for output in op.outputs}
        self.__tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores', 'detection_masks']:
            tensor_name = self.graph_prefix + key + ':0'
            if tensor_name in tensor_names:
                self.__tensor_dict[key] = tf_default_graph.get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in self.__tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(self.__tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(self.__tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(self.__tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, self.__image_shape[0], self.__image_shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            self.__tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            ret, image_np = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.image_np = image_np
                self.__session_runner.add_job(self.__job())
            else:
                self.__in_pipe.wait()

    def __job(self):
        output_dict = self.__tf_sess.run(
            self.__tensor_dict, feed_dict={self.__image_tensor: np.expand_dims(self.image_np, axis=0)})
        self.__out_pipe.push((self.image_np, output_dict))
