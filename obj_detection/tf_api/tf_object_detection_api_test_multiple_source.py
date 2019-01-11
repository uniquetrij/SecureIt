from threading import Thread
from time import sleep

import cv2
from data.obj_tracking.videos import path as videos_path
from flask_movie.flask_movie_api import FlaskMovieAPI

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe

# cap0 = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(1)

session_runner = SessionRunner()
detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, None, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()
session_runner.start()
detection.run()

def detect_objects(cap, pipe, use):
    if not use:
        ret_pipe = Pipe()
    else:
        ret_pipe = None

    while True:
        ret, image = cap.read()
        if not ret:
            continue
        inference = Inference(image.copy(), return_pipe=ret_pipe)
        detector_ip.push(inference)
        if not use:
            ret_pipe.pull_wait()
            ret, inference = ret_pipe.pull(True)
        else:
            detector_op.pull_wait()
            ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            pipe.push(i_dets.get_annotated())
        sleep(0.1)

if __name__ == '__main__':
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    # cap2 = cv2.VideoCapture(2)

    pipe0 = Pipe(limit=1)
    pipe1 = Pipe(limit=1)
    # pipe2 = Pipe(limit=1)


    fs = FlaskMovieAPI()
    Thread(target=fs.get_app().run, args=("0.0.0.0",)).start()
    fs.create('store_feed', pipe0)
    fs.create('shelf_feed', pipe1)
    # fs.create('stock_feed', pipe2)

    Thread(target=detect_objects, args=(cap0, pipe0, False,)).start()
    Thread(target=detect_objects, args=(cap1, pipe1, True,)).start()
    # Thread(target=detect_objects, args=(cap2, pipe2,)).start()

