#!/usr/bin/env python
import multiprocessing
import uuid
from multiprocessing import Process
from uuid import UUID

from flask import Flask, jsonify, json, stream_with_context, render_template
from flask_cors import CORS
from flask_restful import Api, Resource
import matplotlib.path as mplPath




from flask import Response
from threading import Thread
from time import sleep
import cv2

from tf_session.tf_session_utils import Pipe

cap = cv2.VideoCapture(-1)

pipe = Pipe()


def load():
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames Per Second:", fps, "\n")
    while(True):
        ret, image = cap.read()
        if not ret:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except:
                pass
            continue
        pipe.push(image)
        sleep(0.05)

app = Flask(__name__)
CORS(app)

def genVideoFeed():
    Thread(target=load).start()
    """Video streaming generator function."""

    while True:
        try:
            # cv2.imshow("image", image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', pipe.pull(True)[1])[1].tostring() + b'\r\n')
        except GeneratorExit:
            return
        except:
            pass


@app.route('/')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genVideoFeed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__ == '__main__':
    print("in main")
    Thread(target=load).start()

    app.run()
