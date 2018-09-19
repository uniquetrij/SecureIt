
import numpy as np
import cv2
from collections import deque
from scipy.ndimage.measurements import label



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes



def draw_labeled_bboxes(img, labels, roi):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if ((roi[0] <= bbox[0][0] <= roi[2]) and (roi[1] <= bbox[0][1] <= roi[3])) or ((roi[0] <= bbox[1][0] <= roi[2]) and (roi[1] <= bbox[1][1] <= roi[3])):
            cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 2)
            cv2.putText(img,"Trespassing", (30,40), cv2.FONT_HERSHEY_PLAIN,2, 255)
    # Return the image
    return img




dq=deque(maxlen=8)
def heat_img_video(image,bboxes, region_of_interest):

    global heat_list
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,bboxes)

    dq.append(heat)
    avg_heat=np.sum(dq,axis=0)

    # Visualize the heatmap when displaying
    heatmap = np.clip(avg_heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels,region_of_interest)

    return draw_img


def compare_images(image1,image2):

    difference = cv2.subtract(image1, image2)
    size = image1.shape[0] * image1.shape[1] * 3
    perc = 0
    if size>0:
        perc = (((difference<=5).sum())/size) * 100
    if perc>85:

        return [True,perc]
    else:
        return [False,perc]




flag = True
dq1 = deque(maxlen=20)
dq2 = deque(maxlen=20)

cur_image = []
cur_boxes = []
prev_image = []
prev_boxes = []


def Trespassing(frame, region_of_interest):
    global flag
    global dq1
    global dq2
    global cur_image
    global cur_boxes
    global prev_image
    global prev_boxes
    if len(region_of_interest) == 0:
        region_of_interest = (0,0,640,360)
    resized_image = cv2.resize(frame, (640, 360))
    resized_image = cv2.rectangle(resized_image, (region_of_interest[0], region_of_interest[1]), (region_of_interest[2], region_of_interest[3]), (0,0,255), 2)
    if flag:
        prev_image = resized_image
        prev_boxes = cur_boxes
        cur_image = resized_image
        cur_boxes  = cur_boxes
        flag = False
    else:
        prev_image = cur_image
        prev_boxes = cur_boxes
        cur_image = resized_image

    dq1.append(prev_image)
    dq2.append(prev_boxes)

    fgmask = fgbg.apply(cur_image)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask,kernel,iterations = 1)

    im2, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cur_boxes = []

    for c in contours:
        if cv2.contourArea(c) <  400:
            continue

        else:
            x, y, w, h = cv2.boundingRect(c)
            if y>50 or (y<50 and y+h>100):
                if y>50:
                    y = y-10
                    cur_boxes.append([[x,y],[x + w, y + h + 20]])
                else:
                    cur_boxes.append([[x,y],[x + w, y + h + 10]])

    if len(cur_boxes) is not 0:

        image = heat_img_video(cur_image, cur_boxes, region_of_interest)

    else:
        res = []

        if len(dq1) is 20:

            for i in range(0,len(dq1),19):

                prev_image = dq1[i]

                for box in dq2[i]:
                    img1 = prev_image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                    img2 = cur_image[box[0][1]:box[1][1], box[0][0]:box[1][0]]

                    res.append(compare_images(img1,img2))
            if len(res) is 2:

                if res[0][0] or res[1][0]:

                    avg = (res[0][1]+res[1][1])/2
                    if avg > 75 and avg<100:
                        cur_boxes = dq2[19]
                        image = heat_img_video(cur_image, cur_boxes, region_of_interest)

                    else:
                        image = cur_image
                        cur_boxes =[]
                        dq.clear()
                else:
                    image = cur_image
                    cur_boxes =[]
                    dq.clear()
            else:
                image = cur_image
                cur_boxes =[]
                dq.clear()
        else:

            for box in prev_boxes:
                img1 = prev_image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                img2 = cur_image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                similar,perc = compare_images(img1,img2)

                if similar:
                    cur_boxes.append(box)
            if len(cur_boxes) > 0:
                image = heat_img_video(cur_image, cur_boxes, region_of_interest)

            else:
                image = cur_image
                cur_boxes = []
                dq.clear()
    return image



cap = cv2.VideoCapture('/home/developer/PycharmProjects/SecureIt/data/videos/trespassing_detection/final.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
arrays = []
# region_of_interest = (0, 125, 400, 300)
region_of_interest = ()
while True:

    ret, frame = cap.read()

    if ret:
        arrays.append(frame)
        image = Trespassing(frame, region_of_interest)
        cv2.imshow('frame',image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()




