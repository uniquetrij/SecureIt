from threading import Thread
from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/data/videos/People Counting Demonstration.mp4")
if __name__ == '__main__':
    tfSession = SessionRunner()
    while True:
        ret, image = cap.read()
        if ret:
            break

    detection = TFObjectDetectionAPI(tfSession, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', False)
    ip = detection.get_in_pipe()
    op = detection.get_out_pipe()

    tfSession.start()
    detection.run()


    # for i in range(1000):

    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())

        ret, inference = op.pull()
        if ret:
            # print(ret)
            cv2.imshow("", TFObjectDetectionAPI.annotate(inference))
            cv2.waitKey(1)
        else:
            op.wait()

    tfSession.stop()
