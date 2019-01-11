import cv2
from flask import Flask, Response
from flask_cors import CORS

class FlaskMovieAPI:

    def __init__(self, app=None):
        if not app:
            app = Flask(__name__)
        self.__app = app
        self.__routes = {}
        CORS(app)
        @self.__app.route('/<route>')
        def video_feed(route):
            return Response(self.__generate(self.__routes[route]), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        self.__routes[route]=pipe

    def get_app(self):
        return self.__app
