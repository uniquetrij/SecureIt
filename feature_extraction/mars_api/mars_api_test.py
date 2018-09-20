import cv2

from feature_extraction.mars_api.mars_api import MarsExtractorAPI
from tf_session.tf_session_runner import SessionRunner
import numpy as np
import tensorflow as tf

from tf_session.tf_session_utils import Pipe, Inference

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture("/home/developer/PycharmProjects/SecureIt/data/videos/People Counting Demonstration.mp4")
if __name__ == '__main__':
    session_runner = SessionRunner()
    while True:
        ret, image = cap.read()
        if ret:
            break
    image_shape = (224, 224, 3)

    session = tf.Session()

    image = cv2.resize(image, tuple(image_shape[:2][::-1]))
    image = np.expand_dims(image, axis=0)

    # K.set_session(session)

    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    # op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)

    session_runner.start()
    extractor.run()

    ret_pipe = Pipe()

    # for i in range(1000):

    i = 0
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(Inference(image,ret_pipe,{}))

        ret, feature_inference = ret_pipe.pull()
        if ret:
            print(feature_inference.get_result().shape)
        else:
            ret_pipe.wait()

    session_runner.stop()