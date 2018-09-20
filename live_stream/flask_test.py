#!/usr/bin/env python
import multiprocessing
from threading import Thread
from time import sleep

from flask import Flask
from flask_cors import CORS

from tf_session.tf_session_utils import Pipe

app_flask = Flask(__name__)

CORS(app_flask)

from flask import Response

manager = multiprocessing.Manager()
return_dict = manager.dict()

@app_flask.route('/')
def index():
    """Video streaming home page."""
    # return render_template('index.html')

in_pipe = Pipe()

images = []


from socket import *
import cv2

# host = "34.208.106.39"
host = 'localhost'
port = 5005
buf = 1024
addr = (host, port)
fName = 'img_1.jpg'
timeOut = 0.05

# cap = cv2.VideoCapture(-1)






def foo():
    while True:
        s = socket(AF_INET, SOCK_DGRAM)
        s.bind(addr)

        f = open(fName, 'wb')

        data, address = s.recvfrom(buf)

        try:
            while(data):
                f.write(data)
                s.settimeout(timeOut)
                data, address = s.recvfrom(buf)
        except timeout:
            f.close()
            s.close()
        image = cv2.imread(fName)
        images.append(image)


def genVideoFeed():
    global inference
    """Video streaming generator function."""

    while True:
        sleep(0.025)
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg',images.pop(0))[1].tostring() + b'\r\n')
        except GeneratorExit:
            return
        except:
            pass


@app_flask.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genVideoFeed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    Thread(target=foo).start()
    app_flask.run(port=port)
