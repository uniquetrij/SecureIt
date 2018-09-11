import numpy as np


class Inference:
    def __init__(self, image, num_detections, boxes, classes, scores, masks):
        self.num_detections = int(np.squeeze(num_detections))
        self.image = image
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.masks = masks
        self.height, self.width = image.shape[0], image.shape[1]
        self.denorm_boxes = None
        self.xywh = None

    def get_boxes_normalized(self):
        return self.boxes

    def get_boxes(self):
        if self.denorm_boxes is None:
            self.denorm_boxes = []
            for i in range(self.num_detections):
                self.denorm_boxes.append(self.boxes[i] * [self.height, self.width, self.height, self.width])
            self.denorm_boxes = np.array(self.denorm_boxes)
        return self.denorm_boxes

    def get_boxes_as_xywh(self):
        self.get_boxes()
        if self.xywh is None:
            self.xywh = []
            for i in range(self.num_detections):
                self.xywh.append([self.denorm_boxes[i][1], self.denorm_boxes[i][0],
                                  self.denorm_boxes[i][3] - self.denorm_boxes[i][1],
                                  self.denorm_boxes[i][2] - self.denorm_boxes[i][0]])
            self.xywh = np.array(self.xywh)
        return self.xywh

    def get_scores(self):
        return self.scores[:self.num_detections]

    def get_image(self):
        return self.image
