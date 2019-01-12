import cv2
from flask import Flask, Response
from flask_cors import CORS

class FlaskMovieAPI:

    def __init__(self, app=None):
        if not app:
            app = Flask(__name__)
        self.__app = app
        self.__routes_pipe = {}
        self.__routes_default = {}
        self.__routes_timeout = {}
        CORS(app)
        @self.__app.route('/<route>')
        def video_feed(route):
            return Response(self.__generate(self.__routes_pipe[route],  self.__routes_default[route], self.__routes_timeout[route]), mimetype='multipart/x-mixed-replace; boundary=frame')

    def __generate(self, pipe, default, timeout):
        while True:
            try:
                ret, image = pipe.pull(True)
                if not ret:
                    pipe.pull_wait(timeout)
                    ret, image = pipe.pull(True)
                    if not ret:
                        image = default
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')


            except GeneratorExit:
                print("ERR")
                return
            except:
                print("ERROR")
                pass

    def create(self, route, pipe, default_img=None, timeout=None):
        if timeout is None:
            timeout = 1
        self.__routes_pipe[route]=pipe
        self.__routes_default[route] = default_img
        self.__routes_timeout[route] = timeout

    def get_app(self):
        return self.__app
