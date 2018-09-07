# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(-1)
#
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()



from threading import Thread

import cv2

from tf_session.tf_session_utils import VideoStreamer
from obj_detection.tf_api.ObjectDetection import PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, TFObjectDetectionAPI

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28)
cap = cv2.VideoCapture(-1)

videoStreamer = VideoStreamer()
thread = Thread(target = detector.inferContinuous, args = (videoStreamer, ))
thread.start()

while True:
    ret, image = cap.read()
    if ret:
        videoStreamer.set(image)


cap.release()


detector.inferOnStream(cap)
