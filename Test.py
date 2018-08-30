import numpy as np
from threading import Thread
from time import sleep
import cv2
from tf_api import TFObjectDetectionAPI, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28)

detector.start()

# image = cv2.imread("/home/allahbaksh/PycharmProjects/SecureIt/data/images/people-walking-commercial-drive-landing.jpg")

ip = detector.getInPipe()
op = detector.getOutPipe()

def display():
    while(True):
        ret, inference = op.pull()
        if ret:
            cv2.imshow('TF', cv2.resize(inference.getAnnotatedImage(), (800, 600)))
            print(inference.getCrowdCount())
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        sleep(0.05)

cap = cv2.VideoCapture(-1)
def load():
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames Per Second:", fps, "\n")
    while(True):
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())
        sleep(0.05)

Thread(target=load).start()
Thread(target=display).start()
