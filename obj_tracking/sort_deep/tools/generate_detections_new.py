import colorsys
import os
import errno
import argparse
from threading import Thread

import numpy as np
import cv2
import tensorflow as tf
from matplotlib import axis

from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from obj_tracking.sort_deep.application_util.image_viewer import ImageViewer
from obj_tracking.sort_deep.deep_sort import nn_matching
from obj_tracking.sort_deep.deep_sort.detection import Detection
from obj_tracking.sort_deep.tracking_api import ImageEncoder, PRETRAINED_mars_small128

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28

from obj_tracking.sort_deep.application_util import preprocessing, visualization
from obj_tracking.sort_deep.deep_sort.tracker import Tracker

# cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/obj_tracking/sort_deep/MOT16/train/test.mp4")
cap = cv2.VideoCapture(-1)


# cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/data/videos/People Counting Demonstration.mp4")


from tf_session.tf_session_runner import SessionRunner
session_runner = SessionRunner()
session_runner.start()

while True:
    ret, image = cap.read()
    if ret:
        break

od_api = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28, image.shape, 'tf_api', True)
# od_api = YOLOObjectDetectionAPI('yolo_api', True)
od_ip = od_api.get_in_pipe()
od_op = od_api.get_out_pipe()
od_api.use_session_runner(session_runner)
od_api.run()

ds_api = ImageEncoder(session_runner, PRETRAINED_mars_small128, 'ds_api')
ds_ip = ds_api.get_in_pipe()
ds_op = ds_api.get_out_pipe()
ds_api.run()

viewer = None


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def test():
    # print("In Test")
    global viewer
    count = 0
    detections_out = []

    max_cosine_distance = 100
    nn_budget = None

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    output_filename = os.path.join("../resources/detections/MOT16_POI_train/", "%s.npy" % "MOT16-02")
    while not od_op.is_closed():
        detection_list = []
        ret, inference = od_op.pull()
        if not ret:
            continue
        cv2.imwrite("/home/developer/PycharmProjects/SecureIt/obj_tracking/sort_deep/MOT16/train/MOT16-02/img1/" + (
            str(count).zfill(5)) + ".jpg", inference.get_input())
        count += 1
        # print(count)

        classes = inference.get_classes()
        # print(len(classes))
        scores = inference.get_scores()
        # print(len(scores))

        indices = np.where(np.logical_and(classes == 1, scores > 0.5))[0]
        # labels = [inference.get_labels()[i] for i in indices]
        boxes = [inference.get_boxes_as_xywh()[i] for i in indices]
        scores = [inference.get_scores()[i] for i in indices]
        scores = np.asarray(scores)
        scores = scores.reshape(len(scores), 1)

        print(len(classes))
        # print(len(scores))

        try:

            rows = np.concatenate([boxes, scores], axis=1)
        except:
            print("error")
            continue

        features = create_box_encoder()(inference.get_input(), boxes.copy())
        detection_mat = [np.r_[(count + 1, -1, r, -1, -1, -1, f)] for r, f in zip(rows, features)]

        min_confidence = 0.3
        nms_max_overlap = 0.45
        min_height = 0

        for row in detection_mat:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))

        detections = detection_list
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)
        image = inference.get_input().copy()
        _color = (255, 0, 0)
        thickness = 1
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh().astype(np.int)
            pt1 = int(x), int(y)
            pt2 = int(x + w), int(y + h)
            cv2.rectangle(image, pt1, pt2, _color, thickness)
            label = str(track.track_id)
            if label is not None:
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

                center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
                pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                      text_size[0][1]
                cv2.rectangle(image, pt1, pt2, _color, -1)
                cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 255, 255), thickness)

        cv2.imshow("detected", image)
        cv2.waitKey(1)

        detections_out += detection_mat
        np.save(output_filename, np.asarray(detections_out), allow_pickle=False)

    print("writing to file...")

    np.save(output_filename, np.asarray(detections_out), allow_pickle=False)
    print("finished")


def extract_image_patch(image, bbox, patch_shape):
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
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def create_box_encoder():
    image_shape = ds_api.image_shape

    def encoder(image, boxes):
        global viewer
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        # return image_encoder(image_patches, batch_size)

        ds_ip.push(image_patches)

        ret, features = ds_op.pull()
        while not ret:
            ret, features = ds_op.pull()
        return features

    return encoder


def generate_detections(output_dir):
    thread = Thread(target=test)
    thread.start()

    while True:
        # for i in range(1000):
        ret, frame = cap.read()
        if not ret:
            continue

        # cv2.imshow("live", frame)
        # cv2.waitKey(1)
        # print("pushing frame "+(str(i).zfill(5)))
        od_ip.push(cv2.flip(frame, 1))

    od_ip.close()


if __name__ == "__main__":
    generate_detections("../resources/detections/MOT16_POI_train")
