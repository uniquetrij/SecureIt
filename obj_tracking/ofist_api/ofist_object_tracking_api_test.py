from threading import Thread
from time import sleep

import numpy as np
import cv2
from data.videos import path as videos_path
from data.obj_tracking.outputs import path as out_path

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

cap = cv2.VideoCapture(videos_path.get() + '/t_mobile_demo.mp4')
# cap = cv2.VideoCapture(videos_path.get() + '/ra_rafee_cabin_1.mp4')
# cap = cv2.VideoCapture(-1)
seek = 0

while True:
    ret, image = cap.read()
    if ret:
        if seek == 0:
            break
        seek-=1


# detector =  YOLOObjectDetectionAPI('yolo_api', True)
detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_threading()
detector.run()

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True, use_detection_mask=False)
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

def read():


    while True:

        ret, image = cap.read()
        # image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
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

video_writer = VideoWriter(out_path.get()+"/t_mobile_demo_out_4.avi",image.shape[1], image.shape[0], 25)

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

        for trk in trackers:
            if not trk.is_confident():
                continue
            l = len(str(trk.get_id()))
            d = trk.get_bbox()
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 1)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 5 + (10 * l), int(d[1]) + 15), (0, 69, 255),
                          thickness=cv2.FILLED)
            cv2.putText(frame, str(int(trk.get_id())), (int(d[0]) + 2, int(d[1]) + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), thickness=1)
            patches = trk.get_patches()
            cv2.imshow("ID:" + str(trk.get_id()), patches[-1])
            cv2.waitKey(1)


        # # count+=1
        video_writer.write(frame)



        cv2.imshow("output", frame)
        cv2.waitKey(1)
