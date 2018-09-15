from threading import Thread
from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Pipe

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/data/videos/People Counting Demonstration.mp4")
if __name__ == '__main__':
    session_runner = SessionRunner()
    while True:
        ret, image = cap.read()
        if ret:
            break

    detection = YOLOObjectDetectionAPI('yolo_api', False)
    ip = detection.get_in_pipe()
    # op = detection.get_out_pipe()
    detection.use_session_runner(session_runner)

    session_runner.start()
    detection.run()

    ret_pipe = Pipe()

    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push({'image': image.copy(), 'ret_pipe': ret_pipe})

        ret, inference = ret_pipe.pull()
        if ret:
            print(inference.get_classes())
            cv2.imshow("", inference.anotate())
            cv2.waitKey(1)
        else:
            ret_pipe.wait()

    session_runner.stop()
