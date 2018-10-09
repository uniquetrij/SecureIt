import time
from threading import Thread
import cv2
from age_detection_api.age_detection.age_api import AgeDetection
from age_detection_api.age_detection.sort import Sort
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference
import numpy as np


cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture(videos_path.get()+'/Hitman Agent 47 - car chase scene HD.mp4')

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = AgeDetection()
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()
session_runner.start()
detection.run()
tracker = Sort()


frame_no = 0
def read_video():
    # start = time.time()
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))

t = Thread(target=read_video)
t.start()

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
            cv2.imshow("crop"+str(trk.get_id()), trk.get_face())
        frame_no += 1
        cv2.imshow("Final Output", frame)
        cv2.waitKey(1)


