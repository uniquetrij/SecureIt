from threading import Thread
from time import sleep

import cv2
from data.obj_tracking.videos import path as videos_path

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture(videos_path.get()+'/Hitman Agent 47 - car chase scene HD.mp4')
# cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.0.3")

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
# detection.use_threading()
session_runner.start()
detection.run()

def read():
    seq = 0
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        # detector_ip.push(Inference(image.copy()))
        inference = Inference(image.copy())
        inference.seq=seq
        if not detector_ip.push(inference):
            detector_ip.push_wait()
            seq-=1
        else:
            seq+=1


def run():
    while True:
        detector_op.pull_wait()
        ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            frame = i_dets.get_annotated()
            cv2.imshow("annotated", i_dets.get_annotated())
            cv2.waitKey(1)
            # print(inference.seq)

Thread(target=run).start()
Thread(target=read).start()
