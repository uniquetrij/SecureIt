import cv2
from keras.applications import resnet50

from feature_extraction.resnet50_api import ResNet50ExtractorAPI
from tf_session.tf_session_runner import SessionRunner
import numpy as np
from  keras import backend as K
import tensorflow as tf
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

    extractor = ResNet50ExtractorAPI('rn50_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)

    session_runner.start()
    extractor.run()


    # for i in range(1000):
    i = 0
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())

        ret, inference = op.pull()
        if ret:
            print(inference[1].shape)
        else:
            op.wait()

    session_runner.stop()