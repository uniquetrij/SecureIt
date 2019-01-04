import cv2
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
vcap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.251')
while(1):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        # break
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)
