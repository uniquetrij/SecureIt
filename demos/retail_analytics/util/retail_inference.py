import cv2
import numpy as np


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.__xmin = xmin
        self.__ymin = ymin
        self.__xmax = xmax
        self.__ymax = ymax

        self.c = c
        self.__classes = classes

        self.label = -1
        self.score = -1

    def get_classes(self):
        return  self.__classes

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.__classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.__classes[self.get_label()]

        return self.score

    def get_bbox(self):
        return [self.__xmin, self.__ymin, self.__xmax, self.__ymax]


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


class RetailInference(object):

    def __init__(self, image, labels):
        self.__image = image
        self.__bboxes = None
        self.__labels = labels
        self.__annotated = None

    def get_bboxes(self):
        return self.__bboxes

    def get_annotated(self):
        if self.__annotated is None:
            self.__annotated = np.copy(self.__image)
            self.draw_boxes()
        return self.__annotated

    def draw_boxes(self):
        image_h, image_w, _ = self.__annotated.shape
        for box in self.__bboxes:
            xmin, ymin, xmax, ymax = box.get_bbox()
            xmin = int(xmin * image_w)
            ymin = int(ymin * image_h)
            xmax = int(xmax * image_w)
            ymax = int(ymax * image_h)

            cv2.rectangle(self.__annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            # cv2.putText(image,
            #             labels[box.get_label()] + ' ' + str(box.get_score()),
            #             (xmin, ymin - 13),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1e-3 * image_h,
            #             (0,255,0), 2)
            cv2.putText(self.__annotated, self.__labels[box.get_label()], (xmin, ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image_h,
                        (0, 255, 0), 2)

    def decode_netout(self, netout, anchors, nb_class, obj_threshold=0.35, nms_threshold=0.3):
        grid_h, grid_w, nb_box = netout.shape[:3]

        self.__bboxes = []

        # decode the output by the network
        netout[..., 4] = RetailInference.__sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * RetailInference.__softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_threshold

        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row, col, b, 5:]
                    # print(classes)
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row, col, b, :4]

                        x = (col + RetailInference.__sigmoid(x)) / grid_w  # center position, unit: image width
                        y = (row + RetailInference.__sigmoid(y)) / grid_h  # center position, unit: image height
                        w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                        h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                        confidence = netout[row, col, b, 4]

                        box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)
                        # print("box created")
                        self.__bboxes.append(box)

        # suppress non-maximal boxes
        for c in range(nb_class):
            sorted_indices = list(reversed(np.argsort([box.get_classes()[c] for box in self.__bboxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if self.__bboxes[index_i].get_classes()[c] == 0:
                    continue
                else:
                    for j in range(i + 1, len(sorted_indices)):
                        index_j = sorted_indices[j]

                        if RetailInference.__bbox_iou(self.__bboxes[index_i], self.__bboxes[index_j]) >= nms_threshold:
                            self.__bboxes[index_j].get_classes()[c] = 0

        # remove the boxes which are less likely than a obj_threshold
        self.__bboxes = [box for box in self.__bboxes if box.get_score() > obj_threshold]

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def __softmax(x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)

        return e_x / e_x.sum(axis, keepdims=True)


    @staticmethod
    def __bbox_iou(box1, box2):
        bbox1 = box1.get_bbox()
        bbox2 = box2.get_bbox()
        intersect_w = RetailInference.__interval_overlap([bbox1[0], bbox1[2]], [bbox2[0], bbox2[2]])
        intersect_h = RetailInference.__interval_overlap([bbox1[1], bbox1[3]], [bbox2[1], bbox2[3]])

        intersect = intersect_w * intersect_h

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    @staticmethod
    def __interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    @staticmethod
    def __compute_overlap(a, b):
        """
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        return intersection / ua

    @staticmethod
    def __compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
