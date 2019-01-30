from threading import Thread
from time import sleep

import cv2

from flask_movie.flask_movie_api import FlaskMovieAPI

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe

session_runner = SessionRunner()

detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, None, 'tf_api', True)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_session_runner(session_runner)
detector.use_threading()
session_runner.start()
detector.run()


def detect_objects(cap, pipe, default):
    if not default:
        ret_pipe = Pipe()
    else:
        ret_pipe = None

    def start_cam():
        global i
        while True:
            ret, image = cap.read()
            if not ret:
                continue
            inference = Inference(image.copy(), return_pipe=ret_pipe)
            detector_ip.push_wait()
            detector_ip.push(inference)
            i += 1

    Thread(target=start_cam).start()
    while True:
        if not default:
            ret, inference = ret_pipe.pull(True)
            if not ret:
                ret_pipe.pull_wait()
            else:
                ret_pipe.flush()
        else:
            detector_op.pull_wait()
            ret, inference = detector_op.pull(True)
            if not ret:
                detector_op.pull_wait()
            else:
                detector_op.flush()
        if ret:
            i_dets = inference.get_result()
            pipe.push(i_dets.get_annotated())


if __name__ == '__main__':

    fs = FlaskMovieAPI()
    Thread(target=fs.get_app().run, args=("0.0.0.0",)).start()

    cap = []
    pipe = []

    for i in range(2):
        print(i)
        cap.append(cv2.VideoCapture(i))
        pipe.append(Pipe())
        print(i, pipe[i])
        fs.create('feed_'+str(i), pipe[i])
        Thread(target=detect_objects, args=(cap[i], pipe[i], False)).start()
