from threading import Thread

import numpy as np
# import os.path
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io
import time
import cv2
# from inference import Model
# from variable import *
from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
# from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from obj_tracking.sort.sort import Sort
from tf_session.tf_session_runner import SessionRunner

use_dlibTracker = False
# total_time = 0.0
total_frames = 0

# init tracker
tracker = Sort(max_age=50, min_hits=1, use_dlib=use_dlibTracker)  # create instance of the SORT tracker

if use_dlibTracker:
    print("Dlib Correlation tracker activated!")
else:
    print("Kalman tracker activated!")

# init detector
tfSession = SessionRunner()
# cap = cv2.VideoCapture('/home/allahbaksh/Tailgating_detection/videoplayback1')
# cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/obj_tracking/sort_deep/MOT16/train/test.mp4")
cap = cv2.VideoCapture(-1)
while True:
    flag, image = cap.read()
    if flag:
        break

detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
# detection = YOLOObjectDetectionAPI('yolo_api', True)
ip = detection.get_in_pipe()
op = detection.get_out_pipe()
detection.use_session_runner(tfSession)

tfSession.start()
detection.run()

def readvideo():
    while True: 
        re, img = cap.read()
        
        if not re:
            continue

        # get detections
        ip.push(img.copy())
        time.sleep(0.025)
        # cv2.imshow('Video', frame)


# output video
width, height, _ = image.shape
# print(np.int32(height), np.int32(width))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (np.int32(width), np.int32(height)))

Thread(target=readvideo).start()

while True:
    ret, inference = op.pull()
    if ret:
        # print('Frame no: ' + str(total_frames))
        total_frames += 1
        detections = inference.get_boxes(normalized=False)
        # cv2.imshow("annotated",TFObjectDetectionAPI.annotate(inference))
        # cv2.waitKey(0)
        # print(detections)
        frame = inference.get_image()
        classes = inference.get_classes()
        # print(classes)
        person_detections = []
        scores = inference.get_scores()
        for i in range(len(classes)):
            if classes[i] == 1 and scores[i] > .85:
                person_detections.append([detections[i][1], detections[i][0], detections[i][3], detections[i][2]])
        # person_detections = np.array(person_detections)
        start_time = time.time()
        #update tracker
        trackers = tracker.update(person_detections, frame)

        cycle_time = time.time() - start_time
        # total_time += cycle_time
        for d in trackers:
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 0, 0), 2)
            cv2.putText(frame, str(d[4]), (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255))

        cv2.imshow('Output', frame)
        cv2.waitKey(1)
        out.write(frame)
    else:
        op.wait()



# Thread(target=Tracker).start()

