import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from obj_tracking.ofist_api.ofist_object_tracking_api import OFISTObjectTrackingAPI
from tf_session.tf_session_runner import SessionRunner

session_runner = SessionRunner()
session_runner.start()

cap = cv2.VideoCapture(-1)
while True:
    ret, image = cap.read()
    if ret:
        break

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.run()

tracker = OFISTObjectTrackingAPI()
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

while True:
    ret, image = cap.read()
    if not ret:
        continue
    detector_ip.push(image.copy())

    ret, inference = detector_op.pull()
    if ret:
        trk_ip.push(inference)
    else:
        detector_op.wait()
