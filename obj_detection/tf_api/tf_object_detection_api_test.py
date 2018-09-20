from threading import Thread
from time import sleep

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture('/home/developer/Downloads/Hitman Agent 47 - car chase scene HD.mp4')

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)

session_runner.start()
detection.run()

def read_video():
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))
        # detector_op.wait()

Thread(target=read_video).start()
count = 0
while True:
    detector_op.wait()
    ret, inference = detector_op.pull(True)
    if ret:
        i_dets = inference.get_result()
        # print(i_dets.get_masks()[0].shape)
        frame = i_dets.anotate()
        cv2.imshow("", i_dets.anotate())
        cv2.waitKey(1)
        # cv2.imwrite("/home/developer/Desktop/folder/" + (str(count).zfill(5)) + ".jpg", frame)
        count += 1


# Thread(target=detect_objects).start()
