#!/usr/bin/env python
import cv2
import multiprocessing
import uuid
from multiprocessing import Process

import matplotlib.path as mplPath
from flask import Flask
from flask_cors import CORS
from flask_restful import Api, Resource

from data.obj_tracking.videos import path as videos_path
from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

app = Flask(__name__)
CORS(app)
from flask import Response
# import Flask

import numpy as np
from threading import Thread
from time import sleep

# cap = cv2.VideoCapture(videos_path.get() + '/Hitman Agent 47 - car chase scene HD.mp4')
cap = cv2.VideoCapture(-1)
session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
# detection.use_threading()
session_runner.start()
detection.run()

inference = None

manager = multiprocessing.Manager()
return_dict = manager.dict()


def load():
    # print("LOADED")
    while True:
        ret, image = cap.read()
        print(ret)
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))
        # print("#################################")
        sleep(0.05)


def display():
    global inference
    while (True):
        detector_op.wait()
        ret, inference = detector_op.pull(True)
        # print("done----------------------------")


@app.route('/')
def index():
    """Video streaming home page."""
    # return render_template('index.html')


def genVideoFeed():
    global inference
    """Video streaming generator function."""

    while True:
        try:
            # print("*****************************************************************")
            i_dets = inference.get_result()
            frame = i_dets.get_image()
            print(frame.shape)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[
                       1].tostring() + b'\r\n')
        except GeneratorExit:
            print("E1")
            return
        except Exception as e:
            # print(e)
            pass


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genVideoFeed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def genAnnotatedFeed():
    global inference
    """Video streaming generator function."""

    while True:
        try:
            i_dets = inference.get_result()
            frame = i_dets.get_annotated()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[
                       1].tostring() + b'\r\n')
        except GeneratorExit:
            return
        except:
            pass


@app.route('/annotated_feed')
def annotated_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genAnnotatedFeed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# def genQueueFeed():
#     global inference
#     global return_dict
#
#     """Video streaming generator function."""
#
#     while True:
#         try:
#             image = inference.getImage()
#             image = annotateQueue(image, inference.getDecisionInstances(), return_dict.get('trapiz', None))
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')
#         except GeneratorExit:
#             return
#         except:
#             pass
#
#
# @app.route('/queue_feed')
# def queue_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(genQueueFeed(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


def annotateQueue(img, decisionInstances, region=None):
    height, width = img.shape[0], img.shape[1]
    if not region:
        region = [[0., 0.],
                  [width / 2, 0.],
                  [width / 2, height],
                  [0., height]]

    bbPath = mplPath.Path(
        np.array(region))
    red = (0, 0, 255)
    green = (0, 255, 0)
    for i in range(len(decisionInstances)):
        dClass = decisionInstances[i].getClass().getType()
        dScore = decisionInstances[i].getScore()

        if dClass == 'person' and dScore > 0.6:
            box = decisionInstances[i].getBox().getBoundingCoordinates()
            y_tl, x_tl, y_br, x_br = box[0], box[1], box[2], box[3]
            # print(y_tl, x_tl, y_br, x_br)

            path = [[x_tl, y_tl],
                    [x_tl, y_br],
                    [x_br, y_br],
                    [x_br, y_tl]]
            # print(path)

            color = red
            if bbPath.contains_point(((x_br + x_tl) / 2, y_br)):
                color = green

            img = cv2.polylines(img, [np.int32(path)], 1, color, 3)
    return img


api = Api(app)


class crowdcount(Resource):
    def get(self):
        global inference
        return ({"msg": int(inference.getCrowdCount())})


api.add_resource(crowdcount, '/crowd_count')


class roi(Resource):
    def get(self):
        Process(target=from_image, args=(cap.read()[1], return_dict,)).start()
        return ({"msg": "success"})


api.add_resource(roi, '/roi')


def from_image(image, return_dict):
    try:
        grid_interval = 25
        grid_color = (200, 100, 200)
        points = []
        current = [0, 0]
        width = image.shape[1]
        height = image.shape[0]
        img = image.copy()

        c_x = int(width / 2)
        c_y = int(height / 2)

        for i in range(0, c_x + 1, grid_interval):
            cv2.line(img, (i, 0), (i, height), grid_color, 1)
            cv2.line(img, (width - i, 0), (width - i, height), grid_color, 1)

        for i in range(0, c_y + 1, grid_interval):
            cv2.line(img, (0, i), (width, i), grid_color, 1)
            cv2.line(img, (0, height - i), (width, height - i), grid_color, 1)

        def select_point(event, x, y, flags, param):
            current[0] = x
            current[1] = y
            if event == cv2.EVENT_LBUTTONDBLCLK:
                points.append([x, y])

        winname = uuid.uuid4().hex
        print(winname)
        cv2.namedWindow(winname)
        cv2.imshow(winname, image)
        cv2.resizeWindow(winname, 200, 200)
        cv2.setMouseCallback(winname, select_point)
        cv2.moveWindow(winname, 0, 0)

        while True:
            temp_img = img.copy()
            cv2.putText(temp_img, str(current), (current[0] + 20, current[1]), cv2.FONT_HERSHEY_PLAIN, 0.5,
                        (255, 255, 255), 1)
            for point in points:
                cv2.circle(temp_img, (point[0], point[1]), 1, (255, 0, 0), -1)
            cv2.imshow(winname, temp_img)
            k = cv2.waitKey(20) & 0xFF
            if k == 8:
                try:
                    points.pop()
                except:
                    pass
            if k == 27:
                break

        print("Here!!!")
        roi = np.float32(np.array(points.copy()))
        mark = 0.47 * width

        temp_img = image.copy()

        cv2.polylines(temp_img, [np.int32(roi)], 1, (0, 255, 0), 3)
        cv2.imshow(winname, temp_img)

        roi = roi.tolist()
        if roi:
            return_dict["trapiz"] = roi

        while (True):
            k = cv2.waitKey(0)
    except:
        pass


if __name__ == '__main__':
    Thread(target=load).start()
    Thread(target=display).start()

    app.run(port=5000)
