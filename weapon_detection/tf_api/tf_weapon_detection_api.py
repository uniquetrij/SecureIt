import os
import tarfile
from os import path
from os.path import realpath, dirname
from threading import Thread

import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf

from obj_detection.obj_detection_utils import InferedDetections
from obj_detection.tf_api.object_detection.utils import label_map_util
from obj_detection.tf_api.object_detection.utils import ops as utils_ops
from obj_detection.tf_api.object_detection.utils import visualization_utils as vis_util
from tf_session.tf_session_utils import Pipe


class TFWeaponDetectionAPI:

    @staticmethod
    def __get_dir_path():
        return dirname(realpath(__file__))

    @staticmethod
    def __fetch_model_path():
        dir_path = TFWeaponDetectionAPI.__get_dir_path()
        model_path = dir_path + '/pretrained/'
        path_to_frozen_graph = model_path + '/frozen_inference_graph.pb'
        return path_to_frozen_graph

    @staticmethod
    def __fetch_category_indices():
        dir_path = TFWeaponDetectionAPI.__get_dir_path()
        path_to_labels = os.path.join(dir_path + '/data', 'label_map.pbtxt')
        class_count = 1
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=class_count,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        category_dict = {}
        for item in category_index.values():
            category_dict[item['id']] = item['name']
            category_dict[item['name']] = item['id']

        return category_index, category_dict

    def __init__(self, image_shape=None,
                 graph_prefix=None, flush_pipe_on_read=False):
        self.__category_index, self.__category_dict = self.__fetch_category_indices()
        self.__path_to_frozen_graph = self.__fetch_model_path()
        self.__flush_pipe_on_read = flush_pipe_on_read
        self.__image_shape = image_shape

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'

    def __in_pipe_process(self, inference):
        image = inference.get_input()
        data = np.expand_dims(image, axis=0)
        inference.set_data(data)
        return inference

    def __out_pipe_process(self, result):
        result, inference = result
        num_detections = int(result['num_detections'][0])
        detection_classes = result['detection_classes'][0][:num_detections].astype(np.uint8)
        detection_boxes = result['detection_boxes'][0][:num_detections]
        detection_scores = result['detection_scores'][0][:num_detections]
        if 'detection_masks' in result:
            detection_masks = result['detection_masks'][0][:num_detections]
        else:
            detection_masks = None

        result = InferedDetections(inference.get_input(), num_detections, detection_boxes, detection_classes, detection_scores,
                                   masks=detection_masks, is_normalized=True, get_category_fnc=self.get_category,
                                   annotator=self.annotate)
        inference.set_result(result)
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__tf_sess = session_runner.get_session()
        with self.__tf_sess.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.__path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name=self.__graph_prefix)

        tf_default_graph = self.__tf_sess.graph

        self.__image_tensor = tf_default_graph.get_tensor_by_name(self.__graph_prefix + 'image_tensor:0')
        tensor_names = {output.name for op in tf_default_graph.get_operations() for output in op.outputs}
        self.__tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores', 'detection_masks']:
            tensor_name = self.__graph_prefix + key + ':0'
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

            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__session_runner.get_in_pipe().push(
                    (self.__job, inference))
            else:
                self.__in_pipe.wait()

    def __job(self, inference):
        self.__out_pipe.push(
            (self.__tf_sess.run(self.__tensor_dict, feed_dict={self.__image_tensor: inference.get_data()}), inference))

    def get_category(self, category):
        return self.__category_dict[category]

    @staticmethod
    def annotate(inference):
        annotated = inference.image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            annotated,
            inference.get_boxes_tlbr(),
            inference.get_classes().astype(np.int32),
            inference.get_scores(),
            TFWeaponDetectionAPI.__fetch_category_indices()[0],
            instance_masks=inference.get_masks(),
            use_normalized_coordinates=True,
            line_thickness=1)
        return annotated
