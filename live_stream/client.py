from time import sleep

import cv2
from socketIO_client import SocketIO, LoggingNamespace


def processed_frame(json):
    print('processed_frame', len(json['data']))

socketIO = SocketIO('localhost', 8000, LoggingNamespace)
socketIO.on('processed_frame', processed_frame)

cap = cv2.VideoCapture(-1)
# frame = cv2.imread("/home/developer/Downloads/image1.jpeg")
while True:
    image_list = []
    lim = range(10)
    for _ in lim:
        ret, frame = cap.read()
        image_list.append(frame.tolist())
    socketIO.emit('input_frame', {'data': image_list})
    # socketIO.wait(1)
