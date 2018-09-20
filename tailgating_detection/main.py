from threading import Thread

import numpy as np
import time
import cv2

from tailgating_detection.sort import Sort
from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
# from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from tf_session.tf_session_runner import SessionRunner

# total_time = 0.0
from tf_session.tf_session_utils import Inference

total_frames = 0

# init tracker
tracker = Sort(max_age=50, min_hits=1)  # create instance of the SORT tracker


cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture('/home/developer/Downloads/Hitman Agent 47 - car chase scene HD.mp4')
# init detector
session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = YOLOObjectDetectionAPI('yolo_api', True)
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

# output video
width, height, _ = image.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (np.int32(height), np.int32(width)))

Thread(target=read_video).start()

while True:
    detector_op.wait()
    ret, inference = detector_op.pull()
    if ret:
        i_dets = inference.get_result()
        detections = i_dets.get_boxes_tlbr(normalized=False)
        frame = i_dets.get_image()
        classes = i_dets.get_classes()
        person_detections = []
        for i in range(len(classes)):
            if classes[i] == 0:
                person_detections.append([detections[i][1], detections[i][0], detections[i][3], detections[i][2]])
        person_detections = np.array(person_detections)
        start_time = time.time()

        #update tracker
        trackers = tracker.update(person_detections, frame)

        cycle_time = time.time() - start_time
        # total_time += cycle_time
        for d in trackers:
            x1 = int(d[0])
            y1 = int(d[1])
            x2 = int(d[2])
            y2 = int(d[3])
            w = x2 - x1
            h = y2 - y1
            cX = int(x1 + w/2)
            cY = int(y1 + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(frame, "In : " + str(tracker.in_counter), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255))
        cv2.putText(frame, "Out : " + str(tracker.out_counter), (50, 70), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255))

        # cv2.line(frame, (1720, 0), (1520, 1080), (0, 0, 255), 5)

        display_frame = cv2.resize(frame, (1200, 800))

        cv2.imshow('Object Tracking', display_frame)
        cv2.waitKey(1)
        out.write(frame)



