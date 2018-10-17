import time
from threading import Thread
import cv2
from age_detection_api.age_detection.age_api import AgeDetection
from age_detection_api.age_detection.sort import Sort
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe
import numpy as np


class AgeApiRunner(object):

    def __init__(self, session_runner):
        self.__detection = AgeDetection()
        self.__detector_ip = self.__detection.get_in_pipe()
        self.__detector_op = self.__detection.get_out_pipe()
        self.__session_runner = session_runner
        self.__detection.use_session_runner(self.__session_runner)
        self.__detection.use_threading()
        self.__session_runner.start()
        self.__detection.run()
        self.__tracker = Sort()

    def get_detector_ip(self):
        return self.__detector_ip
    def get_detector_op(self):
        return self.__detector_op
    def get_tracker(self):
        return self.__tracker

def read_video(detector_ip, cap):
    # start = time.time()
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))

def push_output(detector_op, tracker, age_in_pipe):
    while True:
        detector_op.wait()
        ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            frame = np.copy(i_dets.get_image())
            trackers = tracker.update(i_dets)
            for trk in trackers:
                bbox = trk.get_state()
                ages = trk.get_ages()
                genders = trk.get_genders()
                ethnicity = trk.get_ethnicity()
                age = sum(ages) / len(ages)
                gender = np.average(np.array(genders), axis=0)
                # print(ethnicity)
                # ethnicity = np.argmax(np.array(ethnicity), axis=0)
                # gender = np.sum()
                # print(ethnicity)
                # break
                frame = i_dets.annotate(frame, bbox, int(age), gender, ethnicity[-1])
            age_in_pipe.push(frame)

def runner(age_in_pipe, session_runner, cam_id=-1):
    cap = cv2.VideoCapture(cam_id)
    while True:
        ret, image = cap.read()
        if ret:
            break
    age_api_runner = AgeApiRunner(session_runner=session_runner)
    print(type(age_api_runner))
    t = Thread(target=read_video, args=(age_api_runner.get_detector_ip(), cap,))
    t.start()
    t1 = Thread(target=push_output, args=(age_api_runner.get_detector_op(), age_api_runner.get_tracker(), age_in_pipe, ))
    t1.start()
    return age_in_pipe