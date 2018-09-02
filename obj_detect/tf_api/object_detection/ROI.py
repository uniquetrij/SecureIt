import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np



def from_image(image, pickle_path=None, grid_interval=25, grid_color=(200, 100, 200)):
    points = []
    current = [0, 0]
    width = image.shape[1]
    height = image.shape[0]
    img = image.copy()

    c_x = int(width / 2)
    c_y = int(height / 2)

    for i in range(0, c_x + 1, grid_interval):
        cv2.line(img, (i, 0), (i, height), grid_color, 1)
        cv2.line(img, (width - i, 0), (width - i, height), grid_color, 1)

    for i in range(0, c_y + 1, grid_interval):
        cv2.line(img, (0, i), (width, i), grid_color, 1)
        cv2.line(img, (0, height - i), (width, height - i), grid_color, 1)

    def select_point(event, x, y, flags, param):
        current[0] = x
        current[1] = y
        if event == cv2.EVENT_LBUTTONDBLCLK:
            points.append([x, y])

    cv2.namedWindow('image')
    cv2.resizeWindow('image', 200, 200)
    cv2.setMouseCallback('image', select_point)
    cv2.moveWindow('image', 0, 0)

    while True:
        temp_img = img.copy()
        cv2.putText(temp_img, str(current), (current[0] + 20, current[1]), cv2.FONT_HERSHEY_PLAIN, 0.5,
                    (255, 255, 255), 1)
        for point in points:
            cv2.circle(temp_img, (point[0], point[1]), 1, (255, 0, 0), -1)
        cv2.imshow('image', temp_img)
        k = cv2.waitKey(20) & 0xFF
        if k == 8:
            try:
                points.pop()
            except:
                pass
        if k == 27:
            break

    trapiz = np.float32(np.array(points.copy()))
    mark = 0.47 * width


    temp_img = image.copy()

    cv2.polylines(temp_img, [np.int32(trapiz)], 1, (0, 255, 0), 3)
    cv2.imshow('image', temp_img)
    cv2.waitKey(200000)
    cv2.destroyAllWindows()

    return trapiz


if __name__ == '__main__':
    cap = cv2.VideoCapture(-1)
    print(from_image(cap.read()[1]))
    cap.release()