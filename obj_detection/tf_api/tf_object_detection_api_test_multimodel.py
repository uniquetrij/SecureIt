from threading import Thread
from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner

cap0 = cv2.VideoCapture(-1)
# cap1 = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/data/videos/People Counting Demonstration.mp4")

if __name__ == '__main__':
    tfSession = SessionRunner()
    tfSession.start()

    while True:
        ret, image0 = cap0.read()
        if ret:
            break


    detection0 = TFObjectDetectionAPI(tfSession, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28, image0.shape, 'tf_api_0', True)
    ip0 = detection0.get_in_pipe()
    op0 = detection0.get_out_pipe()
    detection0.run()


    while True:
        ret, image1 = cap0.read()
        if ret:
            break

    detection1 = TFObjectDetectionAPI(tfSession, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image1.shape, 'tf_api_1', True)
    ip1 = detection1.get_in_pipe()
    op1 = detection1.get_out_pipe()
    detection1.run()

    annotated0 = []
    annotated1 = []

    def play(cap, ip, op, lst):
        i = 0

    detection0 = TFObjectDetectionAPI(tfSession, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image0.shape, 'tf_api_0', True)
    ip0 = detection0.get_in_pipe()
    op0 = detection0.get_out_pipe()


    while True:
        ret, image1 = cap0.read()
        if ret:
            break

    detection1 = TFObjectDetectionAPI(tfSession, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image1.shape, 'tf_api_1', False)
    ip1 = detection1.get_in_pipe()
    op1 = detection1.get_out_pipe()

    detection0.run()
    detection1.run()




    def play(cap, ip, op, lst):
        i = 0
        while True:
            ret, image = cap.read()
            if not ret:
                continue
            ip.push(image.copy())

            ret, inference = op.pull()
            if ret:
                lst.append(inference.get_annotated())
                # cv2.imwrite(frame_name+"/"+str(i).zfill(5)+".jpg", inference.get_annotated())
            #     cv2.imshow(frame_name, inference.get_annotated())
            #     cv2.waitKey(1)
            else:
                op.wait()
            i+=1


    Thread(target=play, args=(cap0, ip0, op0, annotated0)).start()
    Thread(target=play, args=(cap0, ip1, op1, annotated1)).start()

    while True:
        if annotated0:
            cv2.imshow("MASK RCNN", annotated0.pop(0))
            cv2.waitKey(1)
        if annotated1:
            cv2.imshow("Faster RCNN", annotated1.pop(0))
            cv2.waitKey(1)
        sleep(0.01)

