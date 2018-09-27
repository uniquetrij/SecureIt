from threading import Thread
from time import sleep

import numpy as np
import cv2
from data.obj_detection.videos import path as videos_path
from data.obj_tracking.output import path as out_path

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from obj_tracking.ofist_api.ofist_object_tracking_api import OFISTObjectTrackingAPI
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference
from utils.video_writer import VideoWriter

session_runner = SessionRunner()
session_runner.start()

cap = cv2.VideoCapture(videos_path.get() + '/video1.avi')
# cap = cv2.VideoCapture(-1)
while True:
    ret, image = cap.read()
    if ret:
        break

# detector =  YOLOObjectDetectionAPI('yolo_api', True)
detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_threading()
detector.run()

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True)
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

def read():
    count = 0

    while True:
        count+=1
        ret, image = cap.read()
        # if count == 100:
        #     detector_ip.close()
            # print("breaking...")
            # trk_ip.close()
            # break
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))
        # print('waiting')
        detector_op.wait()
        # print('done')
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            trk_ip.push(Inference(i_dets))
        # sleep(0.1)

t = Thread(target=read)
t.start()

video_writer = VideoWriter(out_path.get()+"/video1_out.avi",image.shape[1], image.shape[0], 25)

while True:
    # print(detector_op.is_closed())
    trk_op.wait()
    if trk_ip.is_closed():
        # print("Here")
        video_writer.finish()
        break
    ret, inference = trk_op.pull()
    if ret:
        trackers = inference.get_result()
        frame = inference.get_input().get_image()
        patches = inference.get_data()
        for d in trackers:
            l = len(str(d[4]))
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 1)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 5 + (10 * l), int(d[1]) + 15), (0, 69, 255),
                          thickness=cv2.FILLED)
            cv2.putText(frame, str(int(d[4])), (int(d[0]) + 2, int(d[1]) + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), thickness=1)


        # # count+=1
        # video_writer.write(frame)


        # if patches:
        #     for i, patch in enumerate(patches):
        #         cv2.imshow("patch" + str(i), patch)
        #         cv2.waitKey(1)
        cv2.imshow("output", frame)
        cv2.waitKey(1)
