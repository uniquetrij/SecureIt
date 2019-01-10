import cv2


cap = cv2.VideoCapture(2)
cap1 = cv2.VideoCapture(3)
# cap.open(0)
while True:
    ret, image = cap.read(2)
    ret, image1 = cap1.read()

    if ret:
        print(image.shape)
        cv2.imshow('image from camera 1',image)
        cv2.imshow("image from camera 2", image1)
        cv2.waitKey(1)
        # image = cv2.resize(image, ())
        # cv2.imwrite('/home/developer/workspaces/Angular-Dashboard-master/src/assets/rack_image.jpg', image)
        # break