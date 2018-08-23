import pickle
import struct
import threading
from time import sleep

import cv2
import socket

host = '127.0.0.1'
port = 4037

jpeg_quality = 10
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (host, port)
server.bind(server_address)
server.listen(5)

cap = cv2.VideoCapture('/home/uniquetrij/Downloads/fights_01.mp4')

def handle_client_connection(connection):
    while(True):
        success, frame = cap.read()
        result, buffer = cv2.imencode('.jpg', frame, encode_param)
        # data = pickle.dumps(frame)
        # connection.sendall(struct.pack("L", len(data)) + data)
        size = len(buffer)
        # print(size)
        connection.sendall(size.to_bytes(16,'big'))
        # print(type(buffer))
        connection.sendall(buffer)

while (True):
    print("listening...")
    connection, address = server.accept()
    print('accepted')
    client_handler = threading.Thread(
        target=handle_client_connection,
        args=(connection,)  # without comma you'd get a... TypeError: handle_client_connection() argument after * must be a sequence, not _socketobject
    )
    client_handler.start()



