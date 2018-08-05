import numpy as np
import os
from os import path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
import matplotlib

from objtect import ObjectInstance, ObjectDetectorInterface

matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

sys.path.append('tf_api/object_detection')

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

class TFObjectDetectionAPI(ObjectDetectorInterface):

    objectTypes = [None, 'person']

    def __init__(self, model_name = 'ssd_mobilenet_v1_coco_2017_11_17'):

        if tf.__version__ < '1.4.0':
            raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

        self.model_path = 'object_detection/pretrained/'
        self.model_file = model_name + '.tar.gz'
        self.download_base = 'http://download.tensorflow.org/models/object_detection/'
        self.path_to_frozen_graph = self.model_path + model_name + '/frozen_inference_graph.pb'
        path_to_labels = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
        self.class_count = 90
        if not path.exists(self.path_to_frozen_graph):
            self.__download()

        self.__load()
        self.label_map = label_map_util.load_labelmap(path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.class_count,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def __download(self):
        try:
            os.mkdir(self.model_path)
        except:
            pass
        opener = urllib.request.URLopener()
        opener.retrieve(self.download_base + self.model_file, self.model_path+self.model_file)
        tar_file = tarfile.open(self.model_path+self.model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, self.model_path)

    def __load(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')



    def infer(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        decisionInstances = []
        height, width = image.shape[0], image.shape[1]
        for i in range(len(output_dict['detection_classes'])):
            y_tl, x_tl, y_br, x_br = output_dict['detection_boxes'][i] * [height, width, height, width]
            path = [[x_tl, y_tl],
                    [x_tl, y_br],
                    [x_br, y_br],
                    [x_br, y_tl]]
            try:
                type = self.objectTypes[output_dict['detection_classes'][i]]
            except:
                type = 'unknown'
            decisionInstances.append(ObjectInstance(type,
                                                    output_dict['detection_scores'][i],
                                                    path))
        return decisionInstances








