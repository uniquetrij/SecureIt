from threading import Thread

import cv2
from data.obj_tracking.videos import path as videos_path
from data.images import path as images_path

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture(videos_path.get()+'/Hitman Agent 47 - car chase scene HD.mp4')

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    # ret, image = True, cv2.imread(images_path.get()+"/output.jpg")
    if ret:
        break

detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()
session_runner.start()
detection.run()

frame_no = 0

return_pipe = Pipe()

def start_push():
    while True:
        ret, image = cap.read()
        # ret, image
        # ret, image = True, cv2.imread(images_path.get() + "/output.jpg")
        if not ret:
            continue
        inference = Inference(image.copy(), return_pipe)
        detector_ip.push(inference)
        detector_op.wait()


def start_pull():
    while True:
        ret, inference = return_pipe.pull(True)
        if ret:
            i_dets = inference.get_result()
            frame = i_dets.get_annotated()
            cv2.imwrite(images_path.get() + "/annotated_panaroma.jpg", frame)
            cv2.imshow("annotated", i_dets.get_annotated())
            cv2.waitKey(1)

Thread(target=start_push).start()
Thread(target=start_pull).start()

