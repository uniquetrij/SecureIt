from time import sleep

import cv2

from tf_session.session_runner import SessionRunner
from obj_detect.tf_api.detection_api import TFObjectDetectionAPI, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
if __name__ == '__main__':
    tfSession = SessionRunner()
    detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28)
    ip = detection.getInPipe()
    # op = detection.getOutPipe()
    tfSession.load(detection)
    tfSession.start()

    cap = cv2.VideoCapture(-1)
    while (True):
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())
        sleep(0.05)