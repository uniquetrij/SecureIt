from time import sleep

import cv2

cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/obj_tracking/sort_deep/MOT16/train/MOT16-02.webm")

ret, frame = cap.read()
while ret:
    cv2.imshow("", frame)
    ret, frame = cap.read()
    cv2.waitKey(1)
    sleep(0.05)
