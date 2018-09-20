import pickle
import socket
import struct

import cv2

HOST = 'localhost'
PORT = 5005

cap = cv2.VideoCapture(-1)
tcp_socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket_client.connect((HOST, PORT))

while True:
    ret, frame = cap.read()
    data = pickle.dumps(frame)
    tcp_socket_client.sendall(struct.pack("L", len(data)) + data)
