from threading import Thread
from time import sleep

import cv2

from flask_movie.flask_movie_api import FlaskMovieAPI

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe

def detect_objects(cap, pipe, detector, default):
    if not default:
        ret_pipe = Pipe()
    else:
        ret_pipe = None

    def start_cam():
        while True:
            ret, image = cap.read()
            if not ret:
                continue
            inference = Inference(image.copy(), return_pipe=ret_pipe)
            detector.get_in_pipe().push_wait()
            detector.get_in_pipe().push(inference)
            # sleep(0.05)

    Thread(target=start_cam).start()
    while True:
        if not default:
            ret, inference = ret_pipe.pull(True)
            if not ret:
                ret_pipe.pull_wait()
            else:
                ret_pipe.flush()
        else:
            detector.getOutPipe().pull_wait()
            ret, inference = detector.getOutPipe().pull(True)
        if ret:
            i_dets = inference.get_result()
            pipe.push(i_dets.get_annotated())


if __name__ == '__main__':

    fs = FlaskMovieAPI()
    Thread(target=fs.get_app().run, args=("0.0.0.0",)).start()

    session_runner = {}
    detector = {}

    cap = {}
    pipe = {}

    for i in [0,1,2,5]:
        session_runner[i] = SessionRunner(skip=True)
        session_runner[i].start()

        detector[i] = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, None, 'tf_api', True)
        detector[i].use_session_runner(session_runner[i])
        detector[i].run()

        cap[i]=cv2.VideoCapture(i)
        pipe[i]=Pipe()
        fs.create('feed_'+str(i), pipe[i])
        Thread(target=detect_objects, args=(cap[i], pipe[i], detector[i], False)).start()
