import multiprocessing
import pickle
import socket
import struct
from threading import Thread
from time import sleep

import cv2
from flask import Flask
from flask import Response
from flask_cors import CORS

from tf_session.tf_session_utils import Pipe

HOST = 'localhost'
TCP_PORT = 5005
FLASK_PORT = 5006

app_flask = Flask(__name__)
CORS(app_flask)


@app_flask.route('/')
def index():
    """Video streaming home page."""


tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

tcp_socket.bind((HOST, TCP_PORT))
print('Socket bind complete')
tcp_socket.listen(10)
print('Socket now listening')
conn, addr = tcp_socket.accept()

images = []
data = b''
payload_size = struct.calcsize("L")

pipe = Pipe()


def start_tcp_server():
    global data
    global pipe
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]

        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        pipe.push(frame)


def gen_video_feed():
    global pipe
    """Video streaming generator function."""

    while True:
        pipe.wait()
        ret, image = pipe.pull(True)

        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')
        except GeneratorExit:
            print("error")
            return
        except:
            pass


@app_flask.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    Thread(target=start_tcp_server).start()
    app_flask.run(host=HOST, port=5006)
