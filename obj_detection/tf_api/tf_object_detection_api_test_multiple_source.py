from threading import Thread
from time import sleep

import cv2

from flask_movie.flask_movie_api import FlaskMovieAPI

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference, Pipe


session_runner = SessionRunner()
detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, None, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()
session_runner.start()
detection.run()

sleep(5)

def detect_objects(cap, pipe, default):
    if not default:
        ret_pipe = Pipe(limit=1)
    else:
        ret_pipe = None

    def start_cam():
        while True:
            ret, image = cap.read()
            if not ret:
                continue
            inference = Inference(image.copy(), return_pipe=ret_pipe)
            if not detector_ip.push(inference):
                detector_ip.push_wait()

    Thread(target=start_cam).start()
    ts = 0;
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
            # if inference.ts < ts:
            #     continue
            i_dets = inference.get_result()
            pipe.push(i_dets.get_annotated())
            ts = inference.ts





if __name__ == '__main__':
    cap0 = cv2.VideoCapture(0)
    # cap0 = cv2.VideoCapture()
    # cap0 = cv2.VideoCapture("rtsp://admin:admin123@192.168.0.3")

    pipe0 = Pipe(limit=None)
    # pipe1 = Pipe(limit=1)

    fs = FlaskMovieAPI()
    Thread(target=fs.get_app().run, args=("0.0.0.0",)).start()
    fs.create('feed_0', pipe0)
    # fs.create('shelf_feed', pipe1)


    Thread(target=detect_objects, args=(cap0, pipe0, True,)).start()
    # Thread(target=detect_objects, args=(cap1, pipe1, False,)).start()

