import numpy as np
from obj_detection.tf_api.object_detection.utils import visualization_utils as vis_util


class Inference:
    def __init__(self, image, boxes, scores, classes, num_detections, category_index):
        self.image=image
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.num_detections = num_detections
        self.category_index = category_index

    def get_annotated(self):
        annotated = self.image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            annotated,
            self.boxes,
            self.classes.astype(np.int32),
            self.scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=1)
        return annotated
