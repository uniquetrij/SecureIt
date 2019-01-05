import cv2
cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.2")


while True:
    ret,image = cap.read()
    cv2.imshow("frame",image)
    cv2.waitKey(1)


