



#!/usr/bin/env python
import multiprocessing
from threading import Thread
from time import sleep

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import numpy as np

from tf_session.tf_session_utils import Pipe

app_flask = Flask(__name__)

CORS(app_flask)

from flask import Response

import cv2

manager = multiprocessing.Manager()
return_dict = manager.dict()

@app_flask.route('/')
def index():
    """Video streaming home page."""
    # return render_template('index.html')

in_pipe = Pipe()

images = []

# cap = cv2.VideoCapture(-1)


def genVideoFeed():
    global inference
    """Video streaming generator function."""

    while True:
        sleep(0.1)
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('input_frame')
def process_frame(json):
    frames = json['data']
    for frame in frames:
        images.append(np.array(frame))
    #images.extend(frame)

if __name__ == '__main__':
    Thread(target=app_flask.run).start()
    socketio.run(app, port=8000)


# app = Flask(__name__)
# socketio = SocketIO()

#
# if __name__ == '__main__':
#     socketio.run(app, port=8000, host='0.0.0.0')
#     app.run(port=5000, host='0.0.0.0')

#
