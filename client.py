import pickle
import struct

import cv2
import socket

import numpy as np

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "127.0.0.1"
port = 4037
server_address = (host, port)


client.connect(server_address)

while True:

    size = int.from_bytes(client.recv(16), 'big')

    data = client.recv(4096)
    while len(data) < size:
        data += client.recv(4096)

    # packed_msg_size = data[:payload_size]
    #
    # data = data[payload_size:]
    # msg_size = struct.unpack("L", packed_msg_size)[0]
    #
    # while len(data) < msg_size:
    #     data += client.recv(4096)
    #
    # frame_data = data[:msg_size]
    # data = data[msg_size:]
    #
    # frame = pickle.loads(frame_data)
    # # print (frame.size)

    print(type(data))

    data=np.array(data)

    cv2.imshow('client',cv2.imdecode(data,1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# data = client.recv(4096)
# array = np.frombuffer(data, dtype=np.dtype('uint8'))
#
# img = cv2.imdecode(array, 1)
# cv2.imshow("Image", img)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break


# data, server = sock.recvfrom(65507)
#     print("Fragment size : {}".format(len(data)))
#     if len(data) == 4:
#         # This is a message error sent back by the server
#         if(data == "FAIL"):
#             continue
#     array = np.frombuffer(data, dtype=np.dtype('uint8'))
#     img = cv2.imdecode(array, 1)
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break