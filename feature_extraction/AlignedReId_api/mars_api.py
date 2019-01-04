from threading import Thread

import cv2
import numpy as np
from keras import backend as K
from keras.applications import resnet50
import tensorflow as tf

from tf_session.tf_session_runner import SessionRunnable
from tf_session.tf_session_utils import Pipe

from data.feature_extraction.mars_api.pretrained import path as pretrained_path

class MarsExtractorAPI:

    def __init__(self, graph_prefix='', flush_pipe_on_read=False):
        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__model_path = pretrained_path.get()+"/mars-small128.pb"


        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'

    def get_input_shape(self):
        return  self.__image_shape

    def __preprocess(self, original):
        preprocessed = cv2.resize(original, tuple(self.__image_shape[:2][::-1]))
        # preprocessed = resnet50.preprocess_input(preprocessed)
        return preprocessed

    def __in_pipe_process(self, inference):
        images = inference.get_input()
        if type(images) is not list:
            images = [images]
        data = []
        for img in images:
            data.append(self.__preprocess(img))
        # data = np.array(data)
        inference.set_data(data)
        return inference

    def __out_pipe_process(self, result):
        result, inference = result
        inference.set_result(result)
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__tf_sess = session_runner.get_session()

        with self.__tf_sess.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.__model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name=self.__graph_prefix)

        self.__input_var = tf.get_default_graph().get_tensor_by_name(self.__graph_prefix + "images:0" )
        self.__output_var = tf.get_default_graph().get_tensor_by_name(self.__graph_prefix + "features:0")

        assert len(self.__output_var.get_shape()) == 2
        assert len(self.__input_var.get_shape()) == 4

        self.__feature_dim = self.__output_var.get_shape().as_list()[-1]
        self.__image_shape = self.__input_var.get_shape().as_list()[1:]
        # print(self.__image_shape)

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__session_runner.get_in_pipe().push(SessionRunnable(self.__job, inference))
            else:
                self.__in_pipe.wait()

    def __job(self, inference):
        x = inference.get_data()
        if len(x) > 0:
            self.__out_pipe.push((self.__tf_sess.run(self.__output_var, feed_dict={self.__input_var: x}), inference))
        else:
            self.__out_pipe.push((np.zeros((0, self.__feature_dim), np.float32), inference))
