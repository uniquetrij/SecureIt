import json
import time
from datetime import datetime
from threading import Thread
####
import sys
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

URL = 'https://us-central1-retailanalytics-d6ccf.cloudfunctions.net/api/zonetracking'
session_runner = SessionRunner()
session_runner.start()

# cap = cv2.VideoCapture(videos_path.get() + '/ra_rafee_cabin_1.mp4')
#Camera Id For Tracking and Object Detection
cap = cv2.VideoCapture(1)

img_shape = None
while True:
    ret, image = cap.read()
    if ret:
        # print(image.shape)
        img_shape = image.shape
        cv2.imwrite('../../Angular-Dashboard-master/src/assets/rack_image.jpg', image)
        break

# detector =  YOLOObjectDetectionAPI('yolo_api', True)
detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_threading()
detector.run()

config_path = input_path.get() + "/config.json"
with open(config_path) as config_buffer:
    config = json.load(config_buffer)
retail_yolo_detector = YOLO(input_size=config['model']['input_size'],
                         labels=config['model']['labels'],
                         max_box_per_image=config['model']['max_box_per_image'],
                         anchors=config['model']['anchors'])
retail_yolo_detector.use_session_runner(session_runner)
retail_yolo_detector.use_threading()
yolo_ip = retail_yolo_detector.get_in_pipe()
yolo_op = retail_yolo_detector.get_out_pipe()
yolo_input = Pipe()
retail_yolo_detector.run()

Thread(target=mlt.run).start()
tracking_in_pipe = mlt.tracking_in_pipe
stock_in_pipe = mlt.stock_in_pipe
point_set = mlt.point_set_pipe
zone_pipe = mlt.zone_pipe
age_in_pipe = mlt.age_in_pipe
zone_image_update = mlt.zone_image_update
#Camera Id For Age Detection
age_api_runner.runner(age_in_pipe, session_runner, cam_id=2)

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True, use_detection_mask=False, conf_path=input_path.get() + "/demo_zones_1.csv")
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

retail_an_object = RetailAnalytics()
zone_image_update.wait()
ret, flag = zone_image_update.pull()
if ret and flag:
    while True:
        ret, image = cap.read()
        if ret:
            # print(image.shape)
            img_shape = image.shape
            cv2.imwrite('../../Angular-Dashboard-master/src/assets/rack_image.jpg', image)
            break
point_set.wait()
ret, point_set_dict = point_set.pull()
margin = 100
if ret:
    point_set_dict['point_set_2'] = [[0, 0.6*margin], [img_shape[1]-4.25*margin, 0.6*margin],
                                             [img_shape[1]-4.25*margin, img_shape[0]-0.6*margin], [0, img_shape[0]-0.6*margin]]
    retail_an_object.rack_dict = point_set_dict
    retail_an_object.global_init()
timestamp  = time.time()
zone_detection_in_pipe = Pipe()


def read():
    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = image.copy()
        yolo_input.push(image)
        # print("Image read count: ", count)
        # count+=1
        # image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        detector_ip.push(Inference(image))
        detector_op.wait()
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            infer_idets = Inference(i_dets)
            infer_idets.get_meta_dict()['zone_pipe'] = zone_pipe
            trk_ip.push(infer_idets)


def infer_yolo(timestamp, margin):
    while True:
        yolo_input.wait()
        ret, image = yolo_input.pull(flush=True)
        if not ret:
            continue

        ret, flag = zone_image_update.pull()
        if ret and flag:
            cv2.imwrite('../../Angular-Dashboard-master/src/assets/rack_image.jpg', image)

        inference = Inference(image)
        img_shape = image.shape
        ret, point_set_dict = point_set.pull()
        if ret:
            point_set_dict["point_set_2"] = [[margin, margin], [img_shape[1]-margin, margin],
                                             [img_shape[1]-margin, img_shape[0]-margin], [margin, img_shape[0]-margin]]
            print("updated points")
            retail_an_object.rack_dict = point_set_dict
            retail_an_object.global_init()

        inference.get_meta_dict()['warp_points'] = retail_an_object.rack_dict
        yolo_ip.push(inference)
        yolo_op.wait()
        ret, inference = yolo_op.pull()
        ret1, zones = zone_detection_in_pipe.pull()
        if ret:
            i_dets = inference.get_result()
            image = i_dets.get_annotated()
            if ret1 and not zones:
                boxes = i_dets.get_bboxes()
                image = retail_an_object.print_shelfNo(image)
                image = retail_an_object.misplacedBoxes(boxes, image)
                image = retail_an_object.draw_empty_space(boxes, image)
                flag = retail_an_object.change_of_state()
                # cv2.rectangle(image, (margin, margin), (img_shape[1]-margin, img_shape[0]-margin), (0, 0, 255), 3)

                current_time = time.time()
                elapsed_seconds = (current_time - timestamp)
                # print(elapsed_seconds)
                if (elapsed_seconds > 3):
                    if flag == 1:
                        # pass
                        Thread(target=retail_an_object.postdata).start()
                        timestamp = time.time()
            stock_in_pipe.push(image)
            # cv2.imshow("retail_out", image)
            # cv2.waitKey(1)



t = Thread(target=read)
t.start()

thread1 = Thread(target=infer_yolo, args=(timestamp, margin,))
thread1.start()

video_writer = VideoWriter(out_path.get() + "/t_mobile_demo_out_{}.avi".format(time.strftime('%Y_%m_%d_%H_%M')), image.shape[1], image.shape[0], 25)

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
        # trails = inference.get_meta_dict()['trails']
        zones = []
        payload = []
        for trk in trackers:
            d = trk.get_bbox()
            display = str(int(trk.get_id())) + " " + str([z.get_id() for z in trk.get_trail().get_current_zones()])
            l = len(display)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 3)

            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 10 + (15 * l), int(d[1]) + 35), (0, 69, 255),
                          thickness=cv2.FILLED)

            cv2.putText(frame, display, (int(d[0]) + 2, int(d[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), thickness=3)

            # print(trk.get_id(), [z.get_id() for z in trk.get_trail().get_current_zones()])

            # trail = trails[int(d[4])].get_trail()
            # zone = trail.get_zone()
            # if zone is None and trail :
            #     print()

            trail = trk.get_trail()
            exited_zones = trail.get_exited_zones()
            exit_zone = {}
            for zone in exited_zones:
                exit_time = trail.get_exit(zone.get_id())
                entry_time = trail.get_entry(zone.get_id())
                exit_time_str = datetime.fromtimestamp(entry_time).strftime('%Y:%m:%d:%H:%M:%S')
                entry_time_str = datetime.fromtimestamp(entry_time).strftime('%Y:%m:%d:%H:%M:%S')
                exit_zone['rack_id'] = zone.get_id()
                exit_zone['person_id'] = trk.get_id()
                exit_zone['entry_time'] = entry_time_str
                exit_zone['exit_time'] = exit_time_str
                if exit_time - entry_time > 10:
                    # req = requests.post(URL, json=exit_zone)
                    # print(req)
                    payload.append(exit_zone)
                    print(exit_zone)

            zones.extend(trk.get_trail().get_current_zones())
        zone_detection_in_pipe.push(zones)

        overlay = frame.copy()


        for z in tracker.get_zones():
            cv2.polylines(overlay, [np.int32(z.get_coords())], 1,
                          (0, 255, 255), 2)

        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # # count+=1
        # video_writer.write(frame)

        # if patches:
        #     for i, patch in enumerate(patches):
        #         cv2.imshow("patch" + str(i), patch)
        #         cv2.waitKey(1)
        # cv2.resize(frame, ())
        tracking_in_pipe.push(frame)
        # frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        # cv2.imshow("output", frame)
        # cv2.waitKey(1)
