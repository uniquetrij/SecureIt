#!/usr/bin/env python
from time import sleep

from flask import Flask, render_template, Response
# import Flask
import cv2

from threading import Thread

import cv2
from matplotlib import pyplot as plt

from Utils import VideoStreamer
from objtect import ObjectDetector
from tf_api import PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, TFObjectDetectionAPI

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28)
cap = cv2.VideoCapture(-1)

videoStreamIn = VideoStreamer()
videoStreamOut = VideoStreamer()

thread = Thread(target = detector.inferContinuous, args = (videoStreamIn,videoStreamOut,))
thread.start()

app = Flask(__name__)
# vc = cv2.VideoCapture(-1)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    count = -1;
    while True:

        count += 1

        if count // 10 == 0:
            count = 0
        else:
            continue

        ret, image = cap.read()
        if ret:
            videoStreamIn.set(image)
        print("Hello")
        ret, frame = videoStreamOut.get()
        if not ret:
            print("I'm Here")
            continue
        # rval, frame = vc.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()