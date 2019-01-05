import cv2
import numpy as np
# from data.videos import path as videos_path

count=0
def from_image(image,count):
    try:
        grid_interval = 25
        grid_color = (200, 100, 200)
        points = []
        current = [0, 0]
        width = image.shape[1]
        height = image.shape[0]
        img = image.copy()

        c_x = int(width / 2)
        c_y = int(height / 2)

        # for i in range(0, c_x + 1, grid_interval):
        #     cv2.line(img, (i, 0), (i, height), grid_color, 1)
        #     cv2.line(img, (width - i, 0), (width - i, height), grid_color, 1)
        #
        # for i in range(0, c_y + 1, grid_interval):
        #     cv2.line(img, (0, i), (width, i), grid_color, 1)
        #     cv2.line(img, (0, height - i), (width, height - i), grid_color, 1)

        def select_point(event, x, y, flags, param):
            current[0] = x
            current[1] = y
            # print(x, y)
            if event == cv2.EVENT_LBUTTONDBLCLK:
                print(x,",",y)
                points.append([x, y])

        winname = 'window1'
        # print(winname)
        cv2.namedWindow(winname)
        while True:
            count=count+1
            print("Count: ",count)
            points = []
            cv2.imshow(winname, img)
            cv2.resizeWindow(winname, 200, 200)
            cv2.setMouseCallback(winname, select_point)
            cv2.moveWindow(winname, 0, 0)

            while True:
                temp_img = img.copy()
                cv2.putText(temp_img, str(current), (current[0] + 20, current[1]), cv2.FONT_HERSHEY_PLAIN, 0.5,
                            (255, 255, 255), 1)
                for point in points:
                    cv2.circle(temp_img, (point[0], point[1]), 1, (255, 0, 0), -1)
                cv2.imshow(winname, temp_img)
                k = cv2.waitKey(20) & 0xFF
                if k == 8:
                    try:
                        points.pop()
                    except:
                        pass

                if k == 27:
                    # cv2.destroyAllWindows()
                    break


            print("Here!!!")
            print("creating zones")
            with open("/home/developer/PycharmProjects/SecureIt/data/demos/retail_analytics/inputs/demo_zones.csv",'a') as f:
                text = "Z"+str(count)+","+str(points[0][0])+","+str(points[0][1])+","+str(points[1][0])+","+str(points[1][1])+","+str(points[2][0])+","+str(points[2][1])+","+str(points[3][0])+","+str(points[3][1])+"\n"
                print(text)
                f.write(text)
            print(points)
            roi = np.float32(np.array(points.copy()))
            # mark = 0.47 * width
            # count = count + 1


            # temp_img = image.copy()
            print("Drawing boundaries")
            cv2.polylines(img, [np.int32(roi)], 1, (0, 255, 0), 3)
            # image=temp_img
            print("Showing boundaries")
            # cv2.destroyAllWindows()
            cv2.imshow(winname, img)
            # cv2.waitKey(0)
            # count=count+1

        # roi = roi.tolist()
        #
        # if roi:
        #     return roi
        #
        # while(True):
        #     k = cv2.waitKey(0)
    except:
        pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(-1)
    ret = False
    while not ret:
        ret, frame = cap.read()
    print("from_image")
    # while True:
    print(from_image(frame,0))
