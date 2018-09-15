from threading import Thread
from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
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

    detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
    ip = detection.get_in_pipe()
    # op = detection.get_out_pipe()
    detection.use_session_runner(session_runner)

    session_runner.start()
    detection.run()

    ret_pipe = Pipe()

    # for i in range(1000):
    i = 0
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push({'image':image.copy(), 'ret_pipe':ret_pipe})

        ret, inference = ret_pipe.pull()
        if ret:
            print(inference.get_classes())
            cv2.imshow("", TFObjectDetectionAPI.annotate(inference))
            cv2.waitKey(1)
        else:
            ret_pipe.wait()

    session_runner.stop()
