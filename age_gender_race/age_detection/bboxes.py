#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def draw_bboxes(img_path):
    image = cv2.imread(img_path)
    result = detector.detect_faces(image)
    for face in range(len(result)):
        bounding_box = result[face]
        cv2.rectangle(image,(bounding_box[0]-10, bounding_box[1]-10),
              (bounding_box[0]+bounding_box[2]+10, bounding_box[1] + bounding_box[3]+10),(0,155,255),2)
        cv2.imwrite("drawn"+img_path, image)
    print(result)
    
def return_bboxes(img_path):
    image = cv2.imread(img_path)
    result = detector.detect_faces(image)
    return result


if __name__== "__main__":
    draw_bboxes("aa.jpg")