from threading import Thread

import cv2

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from obj_tracking.ofist_api.ofist_object_tracking_api import OFISTObjectTrackingAPI
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

session_runner = SessionRunner()
session_runner.start()

cap = cv2.VideoCapture('/home/developer/Downloads/shoe_tracking.mp4')
# cap = cv2.VideoCapture(-1)
while True:
    ret, image = cap.read()
    if ret:
        break

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', False)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.run()

tracker = OFISTObjectTrackingAPI(flush_pipe_on_read=True)
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()


def read_video():
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))
        detector_op.wait()
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            trk_ip.push(Inference(i_dets))


#
#
# def run_tracker():
#     while True:
#         trk_op.wait()
#         ret, inference = trk_op.pull()
#         if ret:
#             trackers = inference.get_result()
#             frame = inference.get_input().get_image()
#             for d in trackers:
#                 cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 0, 0), 2)
#                 cv2.putText(frame, str(d[4]), (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255))
#
#             cv2.imshow("", frame)
#             cv2.waitKey(1)
#
#
Thread(target=read_video).start()
# Thread(target=detect_objects).start()
# Thread(target=run_tracker).start()

count = 0
while True:
    # ret, image = cap.read()
    # if not ret:
    #     continue
    # detector_ip.push(Inference(image.copy()))
    # detector_op.wait()
    # ret, inference = detector_op.pull()
    # if ret:
    #     i_dets = inference.get_result()
    #     trk_ip.push(Inference(i_dets))
    trk_op.wait()
    ret, inference = trk_op.pull()
    if ret:
        trackers = inference.get_result()
        frame = inference.get_input().get_image()
        patches = inference.get_data()
        for d in trackers:
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 1)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 15, int(d[1]) + 15), (0, 69, 255), thickness=cv2.FILLED)
            cv2.putText(frame, str(int(d[4])), (int(d[0]) + 2, int(d[1]) + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255),thickness=1)

        # cv2.imshow("diff_img", inference.get_meta_dict()['diff_img'])
        # cv2.imshow("mask", inference.get_meta_dict()['mask'])
        # if patches:
        #     cv2.imshow("patch", patches[0])
        cv2.imshow("output", frame)
        cv2.waitKey(1)
        cv2.imwrite("/home/developer/Desktop/folder/" + (str(count).zfill(5)) + ".jpg", frame)
        count+=1
