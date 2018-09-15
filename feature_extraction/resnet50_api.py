from threading import Thread

import cv2
import numpy as np
from keras import backend as K
from keras.applications import resnet50

from tf_session.tf_session_utils import Pipe


class ResNet50ExtractorAPI:

    def __init__(self, graph_prefix='', flush_pipe_on_read=False):
        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__image_shape = (224, 224, 3)

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'

    def __preprocess(self, original):
        preprocessed = cv2.resize(original, tuple(self.__image_shape[:2][::-1]))
        preprocessed = resnet50.preprocess_input(preprocessed)
        return preprocessed

    def __in_pipe_process(self, images):
        if type(images) is not list:
            images = [images]
        originals = []
        preprocessed = []
        for img in images:
            original = img.copy()
            originals.append(original)
            preprocessed.append(self.__preprocess(original))
        return (originals, preprocessed)

    def __out_pipe_process(self, inference):
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        K.set_session(session_runner.get_session())
        self.__tf_sess = K.get_session()

        with self.__tf_sess.as_default():
            self.__model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            ret, data = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__session_runner.add_job(self.__job, data)

            else:
                self.__in_pipe.wait()

    def __job(self, data):
        f_vecs = self.__model.predict(np.array(data[1]))
        print(f_vecs.shape)
        self.__out_pipe.push((data[0], f_vecs))
