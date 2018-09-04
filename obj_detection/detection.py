import numpy as np
from obj_detection.tf_api.object_detection.utils import visualization_utils as vis_util


class Inference:
    def __init__(self, image, boxes, scores, classes, num_detections, category_index):
        self.image = image
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.num_detections = int(num_detections)
        self.category_index = category_index
        self.height, self.width = image.shape[0], image.shape[1]
        self.denorm_boxes = None
        self.xywh = None
        # self.y_tl, self.x_tl, self.y_br, self.x_br = boxes * [self.height, self.width, self.height, self.width]

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

    def get_boxes(self):
        return self.boxes

    def get_denorm_boxes(self):
        if self.denorm_boxes is None:
            self.denorm_boxes = []
            for i in range(self.num_detections):
                self.denorm_boxes.append(self.boxes[i] * [self.height, self.width, self.height, self.width])
            self.denorm_boxes = np.array(self.denorm_boxes)
        return self.denorm_boxes

    def get_boxes_as_xywh(self):
        self.get_denorm_boxes()
        if self.xywh is None:
            self.xywh = []
            for i in range(self.num_detections):
                self.xywh.append([self.denorm_boxes[i][1], self.denorm_boxes[i][0],self.denorm_boxes[i][3]-self.denorm_boxes[i][1], self.denorm_boxes[i][2]-self.denorm_boxes[i][0]])
            self.xywh = np.array(self.xywh)
        return self.xywh


    def get_scores(self):
        return self.scores[:self.num_detections]

    def get_image(self):
        return self.image


