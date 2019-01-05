import cv2
import os
from data.videos import path as videos_path
from utils.video_writer import VideoWriter

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
vcap = cv2.VideoCapture('rtsp://admin:admin123@192.168.1.2')
video_writer = None
i = 500

while(i>0):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        # break
    else:
        if not video_writer:
            video_writer = VideoWriter(path=videos_path.get() + '/ip_cam.mkv', width=frame.shape[0], height=frame.shape[1], fps=25)
        # cv2.imshow('VIDEO', frame)
        video_writer.write(frame)
        cv2.waitKey(1)
        i-=1


video_writer.finish()
