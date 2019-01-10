import json
import time
from datetime import datetime
from threading import Thread
####
import sys
# import jsonpickle as jsonpickle
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
    cap = cv2.VideoCapture(0)


# Camera Id For Tracking and Object Detection
# cap = cv2.VideoCapture(-1)

# app = Flask(__name__)
# CORS(app)
#
# # person_association = Person_Item_Association()
# processed_zone_camera_feed={}
# stock_in_pipe = Pipe()

## to get image shape only--------
img_shape = None
#
#commenting this out to remove delay
while True:

    ret, image = cap.read()
    print("before if and ret value", ret )
    if ret:
        print(image.shape)
        # image = cv2.resize(image, (640, 480))
        print(image.shape)
        img_shape = image.shape
        break


# <<<<<<< HEAD
#         cv2.imwrite('../../../Angular-Dashboard-master/src/assets/rack_image.jpg', image)
# =======
#         cv2.imwrite('../../../SmartRetailDashboard/src/assets/rack_image.jpg', image)
# >>>>>>> added tracking for products
        # cv2.imwrite('/home/developer/workspaces/Angular-Dashboard-master/src/assets/rack_image.jpg', image)
        # break
##################################################################################################################

# detector =  YOLOObjectDetectionAPI('yolo_api', True)
detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, (640,480), 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_threading()
detector.run()

# config_path = input_path.get() + "/config.json"
# with open(config_path) as config_buffer:
#     config = json.load(config_buffer)
# retail_yolo_detector = YOLO(input_size=config['model']['input_size'],
#                          labels=config['model']['labels'],
#                          max_box_per_image=config['model']['max_box_per_image'],
#                          anchors=config['model']['anchors'])
# retail_yolo_detector.use_session_runner(session_runner)
# retail_yolo_detector.use_threading()
# yolo_ip = retail_yolo_detector.get_in_pipe()
# yolo_op = retail_yolo_detector.get_out_pipe()
# yolo_input = Pipe()
# retail_yolo_detector.run()Z1,242,375,328,380,322,422,234,420
# Z2,343,383,409,384,419,422,349,420

#
# Thread(target=mlt.run).start()
# Thread(target=server.run_flask_server).start()

tracking_in_pipe = mlt.tracking_in_pipe
# stock_in_pipe = mlt.stock_in_pipe
# point_set = mlt.point_set_pipe
zone_pipe = mlt.zone_pipe
# age_in_pipe = mlt.age_in_pipe
# zone_image_update = mlt.zone_image_update
#Camera Id For Age Detection
# <<<<<<< HEADFalse
# age_api_runner.runner(age_in_pipe, session_runner, cam_id=2)
# =======
# age_api_runner.runner(age_in_pipe, session_runner, cam_id=0)
# >>>>>>> added tracking for products

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True, use_detection_mask=False, conf_path=input_path.get() + "/demo_zones.csv")
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()

retail_an_object = RetailAnalytics()
# zone_image_update.wait()
# ret, flag = zone_image_update.pullq
# if ret and flag:
#     while True:
#         ret, image = cap.read()
#         if ret:
#             # print(image.shape)
#             img_shape = image.shape
#             cv2.imwrite('../../../Angular-Dashboard-master/src/assets/rack_image.jpg', image)
#             print("After Write")
#             break
# point_set.wait()
# ret, point_set_dict = point_set.pull()
margin = 100
points = None
# v_stacks = config["global_init"]["v_stack"]
# h_stacks = config["global_init"]["h_stack"]
# if ret:
#     point_set_dict['point_set_2'] = [[0, 0.6*margin], [img_shape[1]-4.25*margin, 0.6*margin],
#                                              [img_shape[1]-4.25*margin, img_shape[0]-0.6*margin], [0, img_shape[0]-0.6*margin]]
#     points = point_set_dict['point_set_2']
#     retail_an_object.rack_dict = point_set_dict
#     retail_an_object.global_init()
# timestamp  = time.time()
# zone_detection_in_pipe = Pipe()
start_time=time.time()
#
def read():
    skip_frames = 999
    skip_flag = False
    start_time = time.time()
    while True:
        global read_count
        read_count += 1
        # server.stock_in_pipe.wait()
        # print("reading from pipe")
        # ret, image = server.stock_in_pipe.pull()
        # print(ret ,"ret value")
        # global start_time


        # while skip_frames > 0 and skip_flag:
        #     cap.grab()
        #     cap.retrieve()
        #     skip_frames -= 1
        #     print("frame skipped")
        #     continue

        ret, image = cap.read()
        try:
            image = cv2.resize(image, (640,480))
        except:
            pass


        # if image is not None:
        #     print(image.shape, "image value")
        if not ret:
            continue
        track_persons(ret,image)
        # print(time.time() - start_time)
        if skip_frames == 1000 and int(time.time() - start_time) > 10:
            print(time.time() - start_time)
            skip_flag = True
        # print('read function time : {} id {}'.format(time.time() - start_time, read_count))
        # image = image.copy()
        # # yolo_input.push(image)
        # # print("Image read count: ", count)
        # # count+=1
        # # image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        # detector_ip.push(Inference(image))
        # detector_op.wait()
        # ret, inference = detector_op.pull(
        # if ret:
        #     i_dets = inference.get_result()
        #     infer_idets = Inference(i_dets)
        #     infer_idets.get_meta_dict()['zone_pipe'] = zone_pipe
        #     trk_ip.push(infer_idets)

def track_persons(ret,image):
        global track_count
        track_count += 1
        image = image.copy()
        # yolo_input.push(ireadmage)
        # print("Image read count: ", count)
        # count+=1
        # image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        start_time = time.time()
        detector_ip.push(Inference(image))
        detector_op.wait()
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            infer_idets = Inference(i_dets)
            infer_idets.get_meta_dict()['zone_pipe'] = zone_pipe
            trk_ip.push(infer_idets)
            # print('track function time : {} id {}'.format(time.time() - start_time, track_count) )


# def infer_yolo(timestamp, margin, points):
#     while True:
#         yolo_input.wait()
#         ret, image = yolo_input.pull(flush=True)
#         if not ret:
#             continue
#
#         # ret, flag = zone_image_update.pull()
#         # if ret and flag:
#         #     cv2.imwrite('../../../Angular-Dashboard-master/src/assets/rack_image.jpg', image)
#         #     print("After Update")
#
#         inference = Inference(image)
#         img_shape = image.shape
#         ret, point_set_dict = point_set.pull()
#         if ret:
#             point_set_dict["point_set_2"] = [[0, 0.6*margin], [img_shape[1]-4.25*margin, 0.6*margin],
#                                              [img_shape[1]-4.25*margin, img_shape[0]-0.6*margin], [0, img_shape[0]-0.6*margin]]
#             print("updated points")
#             points = point_set_dict['point_set_2']
#             retail_an_object.rack_dict = point_set_dict
#             retail_an_object.global_init()
#
#         inference.get_meta_dict()['warp_points'] = retail_an_object.rack_dict
#         yolo_ip.push(inference)
#         yolo_op.wait()
#         ret, inference = yolo_op.pull()
#         ret1, zones = zone_detection_in_pipe.pull()
#         if ret:
#             i_dets = inference.get_result()
#             image = i_dets.get_annotated()
#             if ret1 and not zones:
#                 boxes = i_dets.get_bboxes()
#                 image = retail_an_object.print_shelfNo(image)
#                 image = retail_an_object.misplacedBoxes(boxes, image)
#                 image = retail_an_object.draw_empty_space(boxes, image)
#                 flag = retail_an_object.change_of_state()
#                 # cv2.rectangle(image, (margin, margin), (img_shape[1]-margin, img_shape[0]-margin), (0, 0, 255), 3)
#
#                 current_time = time.time()
#                 elapsed_seconds = (current_time - timestamp)
#                 # print(elapsed_seconds)
#                 if (elapsed_seconds > 3):
#                     if flag == 1:
#                         # pass
#                         Thread(target=retail_an_object.postdata).start()
#                         timestamp = time.time()
#
#             # print("left corner: ",points[0][1])
#             image = image[int(points[0][1] - margin/2):int(points[2][1] + margin/2), int(points[0][0]):int(points[2][0] + margin/2), :]
#             stock_in_pipe.push(image)
#             # cv2.imshow("retail_out", image)
#             # cv2.waitKey(1)

#
print("Start read thread....")
t = Thread(target=read)
t.start()
# video_writer = VideoWriter(out_path.get() + "/t_mobile_demo_out_{}.avi".format(time.strftime('%Y_%m_%d_%H_%M')),
#                                 image.shape[1], image.shape[0], 25)

while True:
    # print(detector_op.is_closed())
    # global main_count

    main_count += 1
    start_time = time.time()
    trk_op.wait()
    # if trk_ip.is_closed():
    #     # print("Here")
    #     video_writer.finish()
    #     break
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
            # print("value of d ", d)
            display = str(int(trk.get_id())) + " " + str([z.get_id() for z in trk.get_trail().get_current_zones()])
            l = len(display)
            ''' Saving A frame'''
            img_patch = frame[int(d[1]):int(d[3]), int(d[0]): int(d[2])]
            trk.set_image(img_patch)
            # cv2.imwrite("/home/developer/Desktop/mars_test_dataset/5/patch_" + "_" + str(time.time()) + ".jpg",
            #             img_patch)
            '''end here'''
            # cv2.line(frame, (0,465), (639,465), color=(255,0,0), thickness=3)
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
            # print(len(exited_zones))
            exit_zone = {}
            for zone in exited_zones:
                # print(zone.get_id())
                exit_time = trail.get_exit(zone.get_id())
                entry_time = trail.get_entry(zone.get_id())
                exit_time_str = datetime.fromtimestamp(exit_time).strftime('%Y:%m:%d:%H:%M:%S')
                entry_time_str = datetime.fromtimestamp(entry_time).strftime('%Y:%m:%d:%H:%M:%S')

                exit_zone['rack_id'] = zone.get_id()
                exit_zone['person_id'] = trk.get_id()
                exit_zone['entry_time'] = entry_time_str
                exit_zone['exit_time'] = exit_time_str
                # print(entry_time - exit_time)
                if exit_time - entry_time > 4:
                    # req = requests.post(URL, json=exit_zone)
                    # print(req)
                    payload.append(exit_zone)
                    # print(len(trackers))
                    # print("Exiting : ",exit_zone,time.time())
                    # response = {'{},{}'.format(exit_zone['rack_id'], exit_zone['rack_id'])}
                    # response_pickled = jsonpickle.encode(response)
                    data = {'zone_id':exit_zone['rack_id'],'person_id': exit_zone['person_id']}
                    # requests.post(url='http://localhost:5000/notify_zone_exit', json=data)
                    print(exit_zone['rack_id'], exit_zone['person_id'])
                    print(exit_zone)

            zones.extend(trk.get_trail().get_current_zones())
        # zone_detection_in_pipe.push(zones)

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
        # if patches:
        #     for i, patch in enumerate(patches):
        #         cv2.imshow("patch" + str(i), patch)
        #         cv2.imwrite("/home/developer/Desktop/mars_test_dataset/patch_"+str(i)+"_"+str(time.time())+".jpg",patch)
        #         cv2.waitKey(1)
        tracking_in_pipe.push(frame)
        # frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        cv2.imshow("output", frame)
        # print('main function time : {} id {}'.format(time.time() - start_time, main_count))
        cv2.waitKey(1)

#
# @app.route('/person_camera_feed', methods=['POST'])
# def test():
#     r = request
#     # convert string of image data to uint8
#     nparr = np.fromstring(r.data, np.uint8)
#     # decode image
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     # track_persons(True,img)
#     global stock_in_pipe
#     stock_in_pipe.push(img)
#     # cv2.imshow("frame",img)
#
#     # plt.show()
#     # print(img.shape)
#
#     # cv2.imshow("test",img)
#     # cv2.waitKey(1)
#
#     # do some fancy processing here....
#
#     # build a response dict to send back to client
#     response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
#                 }
#     # encode response using jsonpickle
#     response_pickled = jsonpickle.encode(response)
#
#     return Response(response=response_pickled, status=200, mimetype="application/json")
#
#
# if __name__ == '__main__':
#     print("**********************hello")
#     app.run(host='0.0.0.0', debug=True, use_reloader=False)
#
#     t = Thread(target=read)
#     t.start()
#
#     # thread1 = Thread(target=infer_yolo, args=(timestamp, margin, points,))
#     # thread1.start()
#
#     video_writer = VideoWriter(out_path.get() + "/t_mobile_demo_out_{}.avi".format(time.strftime('%Y_%m_%d_%H_%M')),
#                                image.shape[1], image.shape[0], 25)
#
#