import cv2
cap = cv2.VideoCapture(0)
cap.open(0)
while True:
    ret, image = cap.read()

    if ret:
        print(image.shape)
        cv2.imshow('image',image)
        cv2.waitKey(1)
        # image = cv2.resize(image, ())
        # cv2.imwrite('/home/developer/workspaces/Angular-Dashboard-master/src/assets/rack_image.jpg', image)
        # break