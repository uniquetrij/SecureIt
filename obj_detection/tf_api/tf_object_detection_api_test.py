from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner

cap = cv2.VideoCapture(-1)
if __name__ == '__main__':
    tfSession = SessionRunner()
    while True:
        ret, image = cap.read()
        if ret:
            break

    detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
    ip = detection.get_in_pipe()
    op = detection.get_out_pipe()
    tfSession.load(detection)

    tfSession.start()


    # for i in range(1000):
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())

        ret, inference = op.pull()
        if ret:
            cv2.imshow("", inference.get_annotated())
            cv2.waitKey(1)
        # break

    tfSession.stop()
