import json
import time
from datetime import datetime
from threading import Thread
import sys
from flask import Flask, render_template, Response
import cv2
import os.path
from flask_cors import CORS
from flask import request
from flask import jsonify
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append('../../')
####
from demos.retail_analytics import multithreading as mlt
from age_detection_api.age_detection import age_api_runner

import cv2

import numpy as np
from data.demos.retail_analytics.inputs import path as input_path
from data.demos.retail_analytics.outputs import path as out_path
from data.videos import path as videos_path
from demos.retail_analytics.ra_yolo import YOLO
from demos.retail_analytics.retail_stock_analytics import RetailAnalytics

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from obj_tracking.ofist_api.ofist_object_tracking_api import OFISTObjectTrackingAPI
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe
from utils.video_writer import VideoWriter
from demos.retail_analytics import server
import requests

read_count = 0
track_count = 0
main_count = 0

URL = 'https://us-central1-retailanalytics-d6ccf.cloudfunctions.net/api/zonetracking'
session_runner = SessionRunner()
session_runner.start()

wireless_camera = False
if wireless_camera:
    cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.4")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
else:
    cap = cv2.VideoCapture(3)


img_shape = None
while True:
    ret, image = cap.read()
    # print("before if and ret value", ret )
    if ret:
        # print(image.shape)
        # print(image.shape)
        img_shape = image.shape
        break

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, (640,480), 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_threading()
detector.run()

tracking_in_pipe = mlt.tracking_in_pipe
zone_pipe = mlt.zone_pipe

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True, use_detection_mask=False, conf_path=input_path.get() + "/demo_zones.csv")
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

retail_an_object = RetailAnalytics()
margin = 100
points = None
start_time=time.time()
#
def read():

    while True:
        global read_count
        read_count += 1
        ret, image = cap.read()
        try:
            image = cv2.resize(image, (640,480))
        except:
            pass

        if not ret:
            continue
        start_time = time.time()
        track_persons(ret,image)
        # print(time.time() - start_time)

def track_persons(ret,image):
        global track_count
        track_count += 1
        image = image.copy()
        detector_ip.push(Inference(image))
        start_time = time.time()
        detector_op.wait()
        # print(time.time() - start_time)
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            infer_idets = Inference(i_dets)
            infer_idets.get_meta_dict()['zone_pipe'] = zone_pipe
            trk_ip.push(infer_idets)
            # print('track function time : {} id {}'.format(time.time() - start_time, track_count) )

print("Start read thread....")
t = Thread(target=read)
t.start()

while True:
    main_count += 1
    trk_op.wait()
    # start_time = time.time()
    ret, inference = trk_op.pull()
    if ret:
        trackers = inference.get_result()
        frame = inference.get_input().get_image()
        patches = inference.get_data()
        # print('3 function time' , (time.time() - start_time))
        zones = []
        payload = []
        for trk in trackers:
            d = trk.get_bbox()
            display = str(int(trk.get_id())) + " " + str([z.get_id() for z in trk.get_trail().get_current_zones()])
            l = len(display)
            img_patch = frame[int(d[1]):int(d[3]), int(d[0]): int(d[2])]
            trk.set_image(img_patch)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 3)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 10 + (15 * l), int(d[1]) + 35), (0, 69, 255),thickness=cv2.FILLED)
            cv2.putText(frame, display, (int(d[0]) + 2, int(d[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), thickness=3)
            trail = trk.get_trail()
            exited_zones = trail.get_exited_zones()
            exit_zone = {}
            for zone in exited_zones:
                exit_time = trail.get_exit(zone.get_id())
                entry_time = trail.get_entry(zone.get_id())
                exit_time_str = datetime.fromtimestamp(exit_time).strftime('%Y:%m:%d:%H:%M:%S')
                entry_time_str = datetime.fromtimestamp(entry_time).strftime('%Y:%m:%d:%H:%M:%S')
                exit_zone['rack_id'] = zone.get_id()
                exit_zone['person_id'] = trk.get_id()
                exit_zone['entry_time'] = entry_time_str
                exit_zone['exit_time'] = exit_time_str
                if exit_time - entry_time > 4:
                    payload.append(exit_zone)
                    data = {'zone_id':exit_zone['rack_id'],'person_id': exit_zone['person_id']}
                    print(exit_zone['rack_id'], exit_zone['person_id'])
                    print(exit_zone)
            zones.extend(trk.get_trail().get_current_zones())

        overlay = frame.copy()
        for z in tracker.get_zones():
            cv2.polylines(overlay, [np.int32(z.get_coords())], 1,(0, 255, 255), 2)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        # tracking_in_pipe.push(frame)
        cv2.imshow("output", frame)
        # print('x function time' , (time.time() - start_time))
        cv2.waitKey(1)
