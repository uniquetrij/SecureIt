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
        self.__routes_pipe = {}
        self.__routes_default = {}
        CORS(app)
        @self.__app.route('/<route>')
        def video_feed(route):
            return Response(self.__generate(self.__routes_pipe[route],  self.__routes_default[route]), mimetype='multipart/x-mixed-replace; boundary=frame')

    def __generate(self, pipe, default_img):
        while True:
            try:
                ret, image = pipe.pull(True)
                if not ret:
                    pipe.pull_wait(1)
                    ret, image = pipe.pull(True)
                    if not ret:
                        image = default_img
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')


            except GeneratorExit:
                print("ERR")
                return
            except:
                print("ERROR")
                pass

    def create(self, route, pipe, default_img=None):
        self.__routes_pipe[route]=pipe
        self.__routes_default[route] = default_img

    def get_app(self):
        return self.__app
