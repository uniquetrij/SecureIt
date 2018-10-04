import time
from threading import Thread
from demos.retail_analytics import multithreading as mlt

import cv2
import numpy as np
from data.demos.retail_analytics.inputs import path as input_path
from data.demos.retail_analytics.outputs import path as out_path
from data.videos import path as videos_path

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from obj_tracking.ofist_api.ofist_object_tracking_api import OFISTObjectTrackingAPI
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference
from utils.video_writer import VideoWriter

# from camera_interface.flir_api.flir_camera import FLIRCamera

session_runner = SessionRunner()
session_runner.start()

cap = cv2.VideoCapture(videos_path.get() + '/ra_rafee_cabin_1.mp4')
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

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True, use_detection_mask=False, conf_path=input_path.get() + "/rafee_cabin_zones.csv")
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

Thread(target=mlt.run).start()
stock_image_in = mlt.image_in_pipe
stock_zone_in = mlt.zone_detection_in_pipe

def read():
    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = image.copy()
        stock_image_in.push(image)
        # image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        detector_ip.push(Inference(image))
        detector_op.wait()
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            trk_ip.push(Inference(i_dets))


t = Thread(target=read)
t.start()

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
        for trk in trackers:
            d = trk.get_bbox()
            display = str(int(trk.get_id())) + " " + str([z.get_id() for z in trk.get_trail().get_current_zones()])
            l = len(display)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 1)

            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 5 + (10 * l), int(d[1]) + 15), (0, 69, 255),
                          thickness=cv2.FILLED)

            cv2.putText(frame, display, (int(d[0]) + 2, int(d[1]) + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), thickness=1)

            # print(trk.get_id(), [z.get_id() for z in trk.get_trail().get_current_zones()])

            # trail = trails[int(d[4])].get_trail()
            # zone = trail.get_zone()
            # if zone is None and trail :
            #     print()

            zones.extend(trk.get_trail().get_current_zones())
        stock_zone_in.push(zones)

        overlay = frame.copy()


        for z in tracker.get_zones():
            cv2.polylines(overlay, [np.int32(z.get_coords())], 1,
                          (0, 255, 255), 2)

        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # # count+=1
        # video_writer.write(frame)

        # if patches:
        #     for i, patch in enumerate(patches):
        #         cv2.imshow("patch" + str(i), patch)
        #         cv2.waitKey(1)
        # cv2.resize(frame, ())
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        cv2.imshow("output", frame)
        cv2.waitKey(1)
