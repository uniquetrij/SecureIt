from threading import Thread
from time import sleep

import cv2
from flask import Flask, jsonify, json, stream_with_context, render_template, Response
from flask_cors import CORS

from tf_session.tf_session_utils import Pipe


class FlaskMovieAPI:

    def __init__(self, app=None):
        if not app:
            app = Flask(__name__)
        self.__app = app
        CORS(app)

    def __generate(self, pipe):
        while True:
            try:
                ret, image = pipe.pull(True)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')
            except GeneratorExit:
                return
            except:
                pass

    def create(self, route, pipe):
        @self.__app.route(route)
        def video_feed():
            return Response(self.__generate(pipe), mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_app(self):
        return self.__app


if __name__ == '__main__':
    cap = cv2.VideoCapture(-1)
    pipe = Pipe(limit=1)
    fs = FlaskMovieAPI()
    Thread(target=fs.get_app().run).start()
    fs.create('/',pipe)

    while (True):
        ret, image = cap.read()
        if not ret:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except:
                pass
            continue
        if not pipe.push(image):
            pipe.push_wait()
        sleep(0.05)

