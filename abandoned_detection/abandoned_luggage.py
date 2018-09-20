from threading import Thread
import numpy as np
import tensorflow as tf
import cv2
import time
import math
import copy
from matplotlib import pyplot as plt


from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference


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

    if len(list_bag)>1:
        for i in range(0, len(bags)-1):
            if bags[i] not in list_bag:
                continue
            else:
                for j in range(i+1, len(bags)):
                    iou = bb_intersection_over_union(bags[i],bags[j])
                    if iou>0.69:                  
                        list_bag.remove(bags[j])

    return list_bag
    

def compare_images(image1,image2):

    difference = cv2.subtract(image1, image2)
    size = image1.shape[0] * image1.shape[1] * 3
    perc = 0
    if size>0:
        perc = (((difference<=5).sum())/size) * 100
    return perc
    
    
def get_bag_count(prev_dq_bags, bag):
    
    bag_indices = []
    bag_index_row = 0
    bag_index_col = 0
    cnt = 0

    for i in range(0,len(prev_dq_bags)):

        if bag in prev_dq_bags[i]:
            cnt += 1
            bag_index_row = i
            bag_index_col = list(prev_dq_bags[i]).index(bag)
            bag_indices.append([bag_index_row,bag_index_col])

        else:
            for j in range(0,len(prev_dq_bags[i])):
                iou = bb_intersection_over_union(bag, prev_dq_bags[i][j])
                if iou>=0.25:
                    cnt+=1
                    bag_index_row = i
                    bag_index_col = j
                    bag_indices.append([bag_index_row,bag_index_col])
                    break   
    
    return cnt, bag_index_row, bag_index_col, bag_indices
    
def cal_avg_owner_perc(owner_list, bag, person, cur_image):
    
    count = 0
    perc_sum = 0
    
    for i in range(0,len(owner_list)):
        if owner_list[i] == 0:
            continue
        else:
            
            max_iou = 0
            idx_i = -1
            idx_j = -1
            for j in range(0, len(owner_list[i])):
                
                if bag == owner_list[i][j][0]:
                    idx_i = i
                    idx_j = j
                    break
                else:
                    iou = bb_intersection_over_union(bag, owner_list[i][j][0])
                    if iou >= 0.40:
                        if max_iou < iou:
                            max_iou = iou
                            idx_i = i
                            idx_j = j
            
            if idx_i >= 0 and idx_j >= 0:
                count += 1
                
                x1 = owner_list[idx_i][idx_j][1][0]
                y1 = owner_list[idx_i][idx_j][1][1]
                x2 = owner_list[idx_i][idx_j][1][2]
                y2 = owner_list[idx_i][idx_j][1][3]

                img_1 = owner_list[i][j][2][y1:y2,x1:x2]
                img_2 = cur_image[person[1]:person[3], person[0]:person[2]]

                resized_image = cv2.resize(img_1, (img_2.shape[1], img_2.shape[0])) 
                resized_image_1 = cv2.resize(img_2, (img_2.shape[1], img_2.shape[0])) 

                perc = compare_images(resized_image, resized_image_1)
                perc_sum += perc

    if count == 0:
        return perc_sum
    else:
        return perc_sum/count
    
    
def update_owner(first_60_owner, last_60_owner, bag_indices, bag, persons, cur_image):
    

    if len(bag_indices) <120:
        if first_60_owner[len(bag_indices)] == 0:
            first_60_owner[len(bag_indices)] =[[bag, persons, cur_image]]
        else:
            first_60_owner[len(bag_indices)].append([bag, persons, cur_image])
    else:
        if  120 <= (len(bag_indices)) < 240:
            
            if last_60_owner[(len(bag_indices))%120] == 0:
                last_60_owner[(len(bag_indices))%120] = [[bag, persons, cur_image]]
            else:
                last_60_owner[(len(bag_indices))%120].append([bag, persons, cur_image])
        else:
            idx = len(bag_indices)%120
            temp_idx = -1
            max_iou = 0
            if last_60_owner[idx] != 0:
                for i in range(0, len(last_60_owner[idx])):

                    if bag == last_60_owner[idx][i][0]:
                        temp_idx = i
                        break
                        last_60_owner[idx][i][0] = bag
                        last_60_owner[idx][i][1] = persons
                        last_60_owner[idx][i][2] = cur_image 
                        break
                    else:
                        iou = bb_intersection_over_union(last_60_owner[idx][i][0], bag)
                        if iou > 0.40:
                            if max_iou < iou:
                                max_iou = iou
                                temp_idx = i
                last_60_owner[idx][temp_idx][0] = bag
                last_60_owner[idx][temp_idx][1] = persons
                last_60_owner[idx][temp_idx][2] = cur_image


from collections import deque

prev_dq_bags = deque(maxlen=450)
prev_dq_persons = deque(maxlen=450)
first_60_owner = deque(maxlen=120)
last_60_owner = deque(maxlen=120)

for i in range(0, 120):
    first_60_owner.append(0)
    last_60_owner.append(0)


def draw_obj(cur_image, prev_image, bags, persons):
    global prev_dq_bags
    global prev_dq_persons
    global first_60_owner
    global last_60_owner

    bags_own = []
    person_own = []

    for bag in bags:

        count, index_row, index_col, bag_indices = get_bag_count(prev_dq_bags, bag)

        if count == 0:

            mid_bag_x = (bag[0] + bag[2]) // 2
            mid_bag_y = (bag[1] + bag[3]) // 2

            min_dist = 10000000
            idx = -1

            for i in range(0, len(persons)):

                mid_person_x = (persons[i][0] + persons[i][2]) // 2
                mid_person_y = (persons[i][1] + persons[i][3]) // 2

                dist = math.sqrt(((mid_person_x - mid_bag_x) ** 2) + ((mid_person_y - mid_bag_y) ** 2))

                if dist < 75:
                    if dist <= min_dist:
                        min_dist = dist
                        idx = i

            bags_own.append(bag)

            if idx >= 0:

                person_own.append(persons[idx])
                if first_60_owner[0] == 0:
                    first_60_owner[0] = [[bag, persons[idx], cur_image]]
                else:
                    first_60_owner[0].append([bag, persons[idx], cur_image])

            else:

                person_own.append([])
                cv2.rectangle(cur_image, (bag[0], bag[1]), (bag[2], bag[3]), (0, 0, 255), 2)
                cv2.putText(cur_image, "Bag Owner not found", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255))


        elif count > 0:

            idx = -1
            owner_lost_flag = 0
            temp_index = -1

            if len(bag_indices) >= 60:

                for bag_idx in range(len(bag_indices) - 1, len(bag_indices) - 61, -1):

                    row = bag_indices[bag_idx][0]
                    col = bag_indices[bag_idx][1]
                    if len(prev_dq_persons[row]) == 0:
                        owner_lost_flag += 1
                    elif len(prev_dq_persons[row][col]) == 0:
                        owner_lost_flag += 1
                    else:
                        temp_index = bag_idx

                        break

            if owner_lost_flag >= 60:

                max_avg = -1
                idx = -2
                if len(persons) > 0:
                    for k in range(0, len(persons)):
                        if len(persons[k]) > 0:
                            avg_perc_first_60 = cal_avg_owner_perc(first_60_owner, bag, persons[k], cur_image)
                            avg_perc_last_60 = cal_avg_owner_perc(last_60_owner, bag, persons[k], cur_image)
                            total_avg = (avg_perc_first_60 + avg_perc_last_60) / 2

                            if total_avg > 60:
                                if max_avg < total_avg:
                                    max_avg = total_avg
                                    idx = k

                if max_avg < 0:
                    person_own.append([])
                    cv2.rectangle(cur_image, (bag[0], bag[1]), (bag[2], bag[3]), (0, 0, 255), 2)
                    cv2.putText(cur_image, "Abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                else:
                    bag_x = (bag[0] + bag[2]) // 2
                    bag_y = (bag[1] + bag[3]) // 2
                    pers_x = (persons[idx][0] + persons[idx][2]) // 2
                    pers_y = (persons[idx][1] + persons[idx][3]) // 2
                    person_own.append(persons[idx])
                    cv2.rectangle(cur_image, (bag[0], bag[1]), (bag[2], bag[3]), (0, 0, 255), 2)
                    cv2.putText(cur_image, "abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    cv2.line(cur_image, (bag_x, bag_y), (pers_x, pers_y), (0, 0, 255), 2)
                    cv2.rectangle(cur_image, (persons[idx][0], persons[idx][1]), (persons[idx][2], persons[idx][3]),
                                  (0, 255, 255), 2)
                    cv2.putText(cur_image, "owner", (persons[idx][0], persons[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255))

            else:
                if (len(bag_indices) - temp_index) <= 10:
                    index_row = bag_indices[temp_index][0]
                    index_col = bag_indices[temp_index][1]
                    if (len(prev_dq_persons[index_row])) > 0:
                        if (len(prev_dq_persons[index_row][index_col])) > 0:
                            person_owner_box = prev_dq_persons[index_row][index_col]

                            mid_per_x = (person_owner_box[0] + person_owner_box[2]) // 2
                            mid_per_y = (person_owner_box[1] + person_owner_box[3]) // 2

                            min_dist = 10000000
                            max_perc = 0

                            for i in range(0, len(persons)):

                                mid_person_x = (persons[i][0] + persons[i][2]) // 2
                                mid_person_y = (persons[i][1] + persons[i][3]) // 2

                                img_iou = bb_intersection_over_union(person_owner_box, persons[i])

                                if img_iou > 0.40:

                                    cur_img_person = cur_image[persons[i][1]:persons[i][3], persons[i][0]:persons[i][2]]

                                    prev_img_person = prev_image[person_owner_box[1]:person_owner_box[3],
                                                      person_owner_box[0]:person_owner_box[2]]

                                    resized_image = cv2.resize(prev_img_person,
                                                               (cur_img_person.shape[1], cur_img_person.shape[0]))
                                    resized_image_1 = cv2.resize(cur_img_person,
                                                                 (cur_img_person.shape[1], cur_img_person.shape[0]))

                                    perc = compare_images(resized_image, resized_image_1)

                                    if max_perc < perc and perc >50:
                                        max_perc = perc

                                        idx = i

                else:
                    if (len(prev_dq_persons[index_row])) > 0:
                        if (len(prev_dq_persons[index_row][index_col])) > 0:
                            person_owner_box = prev_dq_persons[index_row][index_col]

                            mid_per_x = (person_owner_box[0] + person_owner_box[2]) // 2
                            mid_per_y = (person_owner_box[1] + person_owner_box[3]) // 2

                            min_dist = 10000000
                            max_perc = 0

                            for i in range(0, len(persons)):

                                mid_person_x = (persons[i][0] + persons[i][2]) // 2
                                mid_person_y = (persons[i][1] + persons[i][3]) // 2

                                img_iou = bb_intersection_over_union(person_owner_box, persons[i])

                                if img_iou > 0.40:

                                    cur_img_person = cur_image[persons[i][1]:persons[i][3], persons[i][0]:persons[i][2]]

                                    prev_img_person = prev_image[person_owner_box[1]:person_owner_box[3],
                                                      person_owner_box[0]:person_owner_box[2]]

                                    resized_image = cv2.resize(prev_img_person,
                                                               (cur_img_person.shape[1], cur_img_person.shape[0]))
                                    resized_image_1 = cv2.resize(cur_img_person,
                                                                 (cur_img_person.shape[1], cur_img_person.shape[0]))

                                    perc = compare_images(resized_image, resized_image_1)

                                    if max_perc < perc:
                                        max_perc = perc
                                        min_dist = math.sqrt(
                                            ((mid_person_x - mid_per_x) ** 2) + ((mid_person_y - mid_per_y) ** 2))
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
                    cv2.rectangle(cur_image, (bag[0], bag[1]), (bag[2], bag[3]), (0, 255, 255), 2)
                    cv2.rectangle(cur_image, (persons[idx][0], persons[idx][1]), (persons[idx][2], persons[idx][3]),
                                  (0, 255, 255), 2)
                    cv2.line(cur_image, (bag_cord_x, bag_cord_y), (per_cord_x, per_cord_y), (0, 255, 255), 2)
                    cv2.putText(cur_image, "Unattended", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255))
                    cv2.putText(cur_image, "Owner", (persons[idx][0], persons[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 69, 255))
                elif bag_dist > 300:
                    cv2.rectangle(cur_image, (bag[0], bag[1]), (bag[2], bag[3]), (0, 255, 255), 2)
                    cv2.rectangle(cur_image, (persons[idx][0], persons[idx][1]), (persons[idx][2], persons[idx][3]),
                                  (0, 255, 255), 2)
                    cv2.line(cur_image, (bag_cord_x, bag_cord_y), (per_cord_x, per_cord_y), (0, 255, 255), 2)
                    cv2.putText(cur_image, "Abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    cv2.putText(cur_image, "owner", (persons[idx][0], persons[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255))
                update_owner(first_60_owner, last_60_owner, bag_indices, bag, persons[idx], cur_image)
            elif idx == -1:
                person_own.append([])
                cv2.rectangle(cur_image, (bag[0], bag[1]), (bag[2], bag[3]), (0, 0, 255), 2)
                cv2.putText(cur_image, "Abandoned", (bag[0], bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

    if len(bags) == 0 and len(persons) > 0:

        if len(prev_dq_persons) > 0:
            person_list = prev_dq_persons[-1]

            for prev_person in person_list:
                if len(prev_person) == 0:
                    continue
                else:
                    prev_per_mid_x = (prev_person[0] + prev_person[2]) // 2
                    prev_per_mid_y = (prev_person[1] + prev_person[3]) // 2
                    min_dist = 10000000
                    for cur_person in persons:
                        cur_per_mid_x = (cur_person[0] + cur_person[2]) // 2
                        cur_per_mid_y = (cur_person[1] + cur_person[3]) // 2

                        dist = math.sqrt(
                            ((cur_per_mid_x - prev_per_mid_x) ** 2) + ((cur_per_mid_y - prev_per_mid_y) ** 2))

                        if dist < min_dist:
                            min_dist = dist
                            per_fin = cur_person

                    if min_dist < 30:

                        bag_fin = prev_dq_bags[-1][prev_dq_persons[-1].index(prev_person)]

                        bags_own.append(bag_fin)
                        person_own.append(per_fin)

                        bag_x = (bag_fin[0] + bag_fin[2]) // 2
                        bag_y = (bag_fin[1] + bag_fin[3]) // 2
                        per_x = (per_fin[0] + per_fin[2]) // 2
                        per_y = (per_fin[1] + per_fin[3]) // 2

                        bag_dist = math.sqrt(((bag_x - per_x) ** 2) + ((bag_y - per_y) ** 2))

                        if 125 <= bag_dist <= 300:
                            cv2.rectangle(cur_image, (per_fin[0], per_fin[1]), (per_fin[2], per_fin[3]), (0, 255, 255),
                                          2)
                            cv2.rectangle(cur_image, (bag_fin[0], bag_fin[1]), (bag_fin[2], bag_fin[3]), (0, 255, 255),
                                          2)
                            cv2.line(cur_image, (bag_x, bag_y), (per_x, per_y), (0, 255, 255), 2)
                            cv2.putText(cur_image, "Unattended", (bag_fin[0], bag_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 69, 255))
                            cv2.putText(cur_image, "Owner", (per_fin[0], per_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 69, 255))
                        elif bag_dist > 300:
                            cv2.rectangle(cur_image, (per_fin[0], per_fin[1]), (per_fin[2], per_fin[3]), (0, 255, 255),
                                          2)
                            cv2.rectangle(cur_image, (bag_fin[0], bag_fin[1]), (bag_fin[2], bag_fin[3]), (0, 255, 255),
                                          2)
                            cv2.line(cur_image, (bag_x, bag_y), (per_x, per_y), (0, 255, 255), 2)
                            cv2.putText(cur_image, "Abandoned", (bag_fin[0], bag_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255))
                            cv2.putText(cur_image, "owner", (per_fin[0], per_fin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255))

                        count, index_row, index_col, bag_indices = get_bag_count(prev_dq_bags, bag_fin)

                        if count > 0:
                            update_owner(first_60_owner, last_60_owner, bag_indices, bag_fin, per_fin, cur_image)

    prev_dq_bags.append(bags_own)
    prev_dq_persons.append(person_own)

    return cur_image

if __name__ == '__main__':

    session_runner = SessionRunner()


    # cap0 = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/data/videos/abandoned_detection/abandoned_luggage.avi")
    # frame_count = cap0.get(cv2.CAP_PROP_FRAME_COUNT)
    cap0 = cv2.VideoCapture(-1)
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

    width = image0.shape[1]
    height = image0.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/home/developer/PycharmProjects/SecureIt/data/videos/abandoned_detection/abandoned_luggage_sol1.avi', fourcc, 30.0, (width, height))
    def readvideo():
        while True:
            re, image = cap0.read()
            if re:
                ip0.push(Inference(image.copy()))
            else:
                break
            time.sleep(0.025)


    Thread(target=readvideo).start()
    flag = False
    prev_image = []
    cur_image = []

    while True:
        ret, inference = op0.pull()

        if ret:
            bags = []
            persons = []
            i_dets = inference.get_result()
            detections = i_dets.get_boxes_tlbr(normalized=False).astype(np.int)
            classes = i_dets.get_classes()
            scores = i_dets.get_scores()
            frame = i_dets.get_image()
            for i in range(len(classes)):
                if classes[i] == i_dets.get_category('person') > scores[i] > .75:
                    persons.append((detections[i][1], detections[i][0], detections[i][3], detections[i][2]))
                if classes[i] in [27, 31, 33] and scores[i] > .45:
                    bags.append((detections[i][1], detections[i][0], detections[i][3], detections[i][2]))

            bags = iou_rem(bags)
            persons = iou_rem(persons)

            if flag :
                prev_image = np.copy(cur_image)
                cur_image = np.copy(i_dets.get_image())

            else:

                cur_image = np.copy(i_dets.get_image())
                flag = True

            final_image = draw_obj(cur_image, prev_image, bags, persons)

            out.write(final_image)

            cv2.imshow("", final_image)
            cv2.waitKey(1)

        else:
            op0.wait()

    cap0.release()
    out.release()
    cv2.destroyAllWindows()

    prev_dq_bags.clear()
    prev_dq_persons.clear()
    first_60_owner.clear()
    last_60_owner.clear()
