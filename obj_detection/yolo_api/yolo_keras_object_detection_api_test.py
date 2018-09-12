from threading import Thread
from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from tf_session.tf_session_runner import SessionRunner

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
    op = detection.get_out_pipe()
    detection.use_session_runner(session_runner)

    session_runner.start()
    detection.run()

    # for i in range(1000):

    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())
        # ip.push(cv2.imread("/home/uniquetrij/PycharmProjects/SecureIt/data/images/2.jpg"))

        ret, inference = op.pull()
        if ret:
            print(inference.get_classes())
            cv2.imshow("", YOLOObjectDetectionAPI.annotate(inference))
            cv2.waitKey(1)
        else:
            op.wait()

    session_runner.stop()
