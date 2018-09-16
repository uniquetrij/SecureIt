from threading import Thread

import numpy as np
import math
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


import time
# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def iou_rem(bags):
    list_bag = list(bags)

    if len(list_bag) > 1:
        for i in range(0, len(bags) - 1):
            if bags[i] not in list_bag:
                continue
            else:
                for j in range(i + 1, len(bags)):
                    iou = bb_intersection_over_union(bags[i], bags[j])
                    if iou > 0.69:
                        list_bag.remove(bags[j])

    return list_bag


def compare_images(image1, image2):
    difference = cv2.subtract(image1, image2)
    size = image1.shape[0] * image1.shape[1] * 3
    perc = 0
    if size > 0:
        perc = (((difference <= 5).sum()) / size) * 100
    return perc


from collections import deque
prev_dq_bags = deque(maxlen=300)
prev_dq_persons = deque(maxlen=300)


def draw_obj(image, prev_image, persons, bags, flag):

    cur_image = image


    bags_own = []
    person_own = []

    for bag in bags:
        bag_indices = []
        index = 0
        index_dq_bag = 0
        cnt = 0
        bag_cnt =0
        iou_cnt = 0
        for l in range(0, len(prev_dq_bags)):

            if bag in prev_dq_bags[l]:
                cnt += 1
                index = l
                index_dq_bag = list(prev_dq_bags[l]).index(bag)
                bag_indices.append([index, index_dq_bag])

            else:
                for k in range(0, len(prev_dq_bags[l])):
                    iou = bb_intersection_over_union(bag, prev_dq_bags[l][k])
                    if iou >= 0.45:
                        cnt += 1
                        index = l
                        index_dq_bag = k
                        bag_indices.append([index, index_dq_bag])
                        break
        if len(prev_dq_bags) - index > 15:

            for cntr in range(0, 20):
                if len(prev_dq_bags[len(prev_dq_bags) - cntr - 1]) == 1:
                    iou = bb_intersection_over_union(prev_dq_bags[len(prev_dq_bags) - cntr - 1][0], bag)
                    if iou > 0.25:
                        bag_cnt += 1
                        iou_cnt += 1
            if bag_cnt == 20 and iou_cnt == 20:

                for idc in range(len(prev_dq_bags) - 20, len(prev_dq_bags)):
                    cnt += 1
                    index = idc
                    index_dq_bag = 0
                    bag_indices.append([index, index_dq_bag])

        if cnt == 0:
            mid_bag_x = (bag[0] + bag[2]) // 2
            mid_bag_y = (bag[1] + bag[3]) // 2

            min_dist = 10000000
            idx = -1
            for i in range(0, len(persons)):

                mid_person_x = (persons[i][0] + persons[i][2]) // 2
                mid_person_y = (persons[i][1] + persons[i][3]) // 2

                dist = math.sqrt(((mid_person_x - mid_bag_x) ** 2) + ((mid_person_y - mid_bag_y) ** 2))

                if dist <= min_dist:
                    min_dist = dist
                    fin_x = mid_person_x
                    fin_y = mid_person_y
                    idx = i
            bags_own.append(bag)
            if idx > 0:
                person_own.append(persons[idx])
            else:
                person_own.append([])
            if min_dist > 125:
                cv2.line(image, (mid_bag_x, mid_bag_y), (fin_x, fin_y), (255, 0, 0), 2)
                cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (255, 255, 0), 2)
                cv2.rectangle(image, (persons[idx][0], persons[idx][1]), (persons[idx][2], persons[idx][3]),
                              (255, 255, 0), 2)
                cv2.putText(image, "Bag", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
                cv2.putText(image, "owner", (persons[idx][0], persons[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0))

        elif cnt > 0:
            bag_dist = 0
            idx = -1

            lost_flag = 0
            if len(bag_indices) < 60:
                for bag_idx in range(len(bag_indices) - 1, -1, -1):

                    if len(prev_dq_persons[bag_idx]) == 0:
                        lost_flag += 1
            else:
                for bag_idx in range(len(bag_indices) - 1, len(bag_indices) - 61, -1):

                    if len(prev_dq_persons[bag_idx]) == 0:
                        lost_flag += 1
            print(lost_flag)
            if lost_flag > 60:
                cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (255, 255, 0), 2)
                cv2.putText(image, "Abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            else:
                if (len(prev_dq_persons[index])) > 0:
                    index = bag_indices[-1][0]
                    index_dq_bag = bag_indices[-1][1]
                    print('----------------------------------')
                    print(index)
                    print(index_dq_bag)
                    person_owner_box = prev_dq_persons[index][index_dq_bag]
                    print(person_owner_box[index],'--------')
                    print(person_owner_box[index][index_dq_bag],'########')


                    mid_per_x = (person_owner_box[0] + person_owner_box[2]) // 2
                    mid_per_y = (person_owner_box[1] + person_owner_box[3]) // 2
                    min_dist = 10000000
                    max_perc = 0

                    for i in range(0, len(persons)):

                        mid_person_x = (persons[i][0] + persons[i][2]) // 2
                        mid_person_y = (persons[i][1] + persons[i][3]) // 2

                        if flag:

                            img_iou = bb_intersection_over_union(person_owner_box, persons[i])
                            if img_iou > 0.40:
                                pre_img = prev_image[person_owner_box[1]:person_owner_box[3],
                                          person_owner_box[0]:person_owner_box[2]]
                                curr_img = cur_image[persons[i][1]:persons[i][3], persons[i][0]:persons[i][2]]

                                resized_image = cv2.resize(pre_img, (curr_img.shape[1], curr_img.shape[0]))
                                resized_image_1 = cv2.resize(curr_img, (curr_img.shape[1], curr_img.shape[0]))

                                perc = compare_images(resized_image, resized_image_1)

                                if max_perc < perc:
                                    max_perc = perc
                                    min_dist = math.sqrt(
                                        ((mid_person_x - mid_per_x) ** 2) + ((mid_person_y - mid_per_y) ** 2))
                                    idx = i
                        else:

                            dist = math.sqrt(((mid_person_x - mid_per_x) ** 2) + ((mid_person_y - mid_per_y) ** 2))
                            if dist < min_dist:
                                min_dist = dist
                                idx = i

            bags_own.append(bag)
            if idx >= 0:
                person_own.append(persons[idx])

                bag_cord_x = (bag[0] + bag[2]) // 2
                bag_cord_y = (bag[1] + bag[3]) // 2
                per_cord_x = (persons[idx][0] + persons[idx][2]) // 2
                per_cord_y = (persons[idx][1] + persons[idx][3]) // 2

                bag_dist = math.sqrt(((bag_cord_x - per_cord_x) ** 2) + ((bag_cord_y - per_cord_y) ** 2))

                if 125 <= bag_dist <= 300:
                    cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (255, 255, 0), 2)
                    cv2.rectangle(image, (persons[idx][0], persons[idx][1]), (persons[idx][2], persons[idx][3]),
                                  (255, 255, 0), 2)
                    cv2.line(image, (bag_cord_x, bag_cord_y), (per_cord_x, per_cord_y), (255, 255, 0), 2)
                    cv2.putText(image, "Unattended", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
                    cv2.putText(image, "Owner", (persons[idx][0], persons[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0))
                elif bag_dist > 300:
                    cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (255, 255, 0), 2)
                    cv2.rectangle(image, (persons[idx][0], persons[idx][1]), (persons[idx][2], persons[idx][3]),
                                  (255, 255, 0), 2)
                    cv2.line(image, (bag_cord_x, bag_cord_y), (per_cord_x, per_cord_y), (255, 255, 0), 2)
                    cv2.putText(image, "Abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                    cv2.putText(image, "owner", (persons[idx][0], persons[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0))

            else:

                cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (255, 0, 0), 2)
                cv2.putText(image, "Abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    if len(bags) == 0 and len(persons) > 0:
        length = len(prev_dq_persons) - 1
        bag_dist = 0
        if length > 0:
            person_list = prev_dq_persons[length]

            for person in person_list:

                mid_x = (person[0] + person[2]) // 2
                mid_y = (person[1] + person[3]) // 2
                min_dist = 10000000
                for pers in persons:
                    mid_x1 = (pers[0] + pers[2]) // 2
                    mid_y1 = (pers[1] + pers[3]) // 2

                    dist = math.sqrt(((mid_x1 - mid_x) ** 2) + ((mid_y1 - mid_y) ** 2))

                    if dist < min_dist:
                        min_dist = dist
                        per_fin = pers

                if min_dist < 10:

                    bag_fin = prev_dq_bags[length][prev_dq_persons[length].index(person)]
                    bags_own.append(bag_fin)
                    person_own.append(per_fin)
                    bag_x = (bag_fin[0] + bag_fin[2]) // 2
                    bag_y = (bag_fin[1] + bag_fin[3]) // 2
                    per_x = (per_fin[0] + per_fin[2]) // 2
                    per_y = (per_fin[1] + per_fin[3]) // 2

                    bag_dist = math.sqrt(((bag_x - per_x) ** 2) + ((bag_y - per_y) ** 2))

                    if 125 <= bag_dist <= 300:
                        cv2.rectangle(image, (per_fin[0], per_fin[1]), (per_fin[2], per_fin[3]), (255, 255, 0), 2)
                        cv2.rectangle(image, (bag_fin[0], bag_fin[1]), (bag_fin[2], bag_fin[3]), (255, 255, 0), 2)
                        cv2.line(image, (bag_x, bag_y), (per_x, per_y), (255, 255, 0), 2)
                        cv2.putText(image, "Unattended", (bag_fin[0], bag_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 0))
                        cv2.putText(image, "Owner", (per_fin[0], per_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 0))
                    elif bag_dist > 300:
                        cv2.rectangle(image, (per_fin[0], per_fin[1]), (per_fin[2], per_fin[3]), (255, 255, 0), 2)
                        cv2.rectangle(image, (bag_fin[0], bag_fin[1]), (bag_fin[2], bag_fin[3]), (255, 255, 0), 2)
                        cv2.line(image, (bag_x, bag_y), (per_x, per_y), (255, 255, 0), 2)
                        cv2.putText(image, "Abandoned", (bag_fin[0], bag_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0))
                        cv2.putText(image, "owner", (per_fin[0], per_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


    prev_dq_bags.append(bags_own)
    prev_dq_persons.append(person_own)

    return image



if __name__ == '__main__':

    session_runner = SessionRunner()


    cap0 = cv2.VideoCapture("/home/developer/Downloads/ssd/test_1.mp4")
    while True:
        ret, image0 = cap0.read()
        if ret:
            break


    detection0 = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image0.shape,
                                      'tf_api_0', False)
    detection0.use_session_runner(session_runner)
    ip0 = detection0.get_in_pipe()
    op0 = detection0.get_out_pipe()
    session_runner.start()
    detection0.run()


    def readvideo():
        while True:
            re, image = cap0.read()
            if re:
                ip0.push(image.copy())
            time.sleep(0.025)


    Thread(target=readvideo).start()
    flag = False
    while True:
        ret, inference = op0.pull()
        persons = []
        bags = []
        prev_image=[]
        if ret:
            detections = inference.get_boxes_tlbr(normalized=False).astype(np.int)
            classes = inference.get_classes()
            scores = inference.get_scores()
            for i in range(len(classes)):
                if classes[i] == inference.get_category('person') and scores[i] >= 0.75:
                    persons.append((detections[i][1], detections[i][0], detections[i][3], detections[i][2]))
                if classes[i] in [27, 31, 33]:
                    print(inference.get_category(classes[i]))
                    bags.append((detections[i][1], detections[i][0], detections[i][3], detections[i][2]))
            print(flag)
            bags = iou_rem(bags)
            persons = iou_rem(persons)
            print(len(bags))
            if flag:
                prev_image = np.copy(cur_image)
                cur_image = np.copy(inference.get_input())

            else:
                cur_image = np.copy(inference.get_input())
                flag = True

            # print(persons[0])

            frame = draw_obj(inference.get_input(), prev_image, persons, bags, flag)
            cv2.imshow("", frame)
            cv2.waitKey(1)
        else:
            op0.wait()

        prev_dq_bags.clear()
        prev_dq_persons.clear()

