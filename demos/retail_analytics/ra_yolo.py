from threading import Thread

import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
from demos.retail_analytics.utilities  import decode_netout
from data.demos.retail_analytics.trained import path as model_path
import tensorflow as tf

from demos.retail_analytics.util.retail_inference import RetailInference
from tf_session.tf_session_runner import SessionRunnable
from tf_session.tf_session_utils import Pipe


class YOLO(object):
    def __init__(self, input_size, labels, max_box_per_image, anchors, flush_pipe_on_read=False):
        self.__flush_pipe_on_read = flush_pipe_on_read
        self.__model_path = model_path.get()
        self.__model = None
        self.__input_size = input_size
        self.__labels = list(labels)
        self.__nb_class = len(self.__labels)
        self.__nb_box = len(anchors) // 2
        self.__class_wt = np.ones(self.__nb_class, dtype='float32')
        self.__anchors = anchors
        self.__max_box_per_image = max_box_per_image
        self.__session_runner = None
        self.__tf_sess = None

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)
        self.__run_session_on_thread = False

    def use_threading(self, run_on_thread=True):
        self.__run_session_on_thread = run_on_thread

    def __normalize(self, image):
        return image / 255.

    def __preprocess(self, original, warp_points):
        img_shape = original.shape
        test1 = np.array(warp_points['point_set_1'], dtype="float32")
        test2 = np.array(warp_points['point_set_2'], dtype="float32")

        # print(test2)
        # points = [warp_points['point_set_2'][0],warp_points['point_set_2'][2]]
        # print(points)
        # np.array([[0, 0], [img_shape[0], 0], [img_shape[0], img_shape[1]], [0, img_shape[1]]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(test1, test2)
        warped_image = cv2.warpPerspective(original, M, (img_shape[1],img_shape[0]),flags=cv2.INTER_LINEAR)


        # warped_image = warped_image[points[0][1]:points[1][1],points[0][0]:points[1][0],:]
        return warped_image

    def __prepare_data(self,image):
        image_h, image_w, _ = image.shape
        preprocessed = cv2.resize(image, (self.__input_size, self.__input_size))
        preprocessed = self.__normalize(preprocessed)

        preprocessed = preprocessed[:, :, ::-1]
        preprocessed = np.expand_dims(preprocessed, 0)
        return preprocessed

    def __in_pipe_process(self, inference):
        image = inference.get_input()
        warp_points = inference.get_meta_dict()['warp_points']
        dummy_array = np.zeros((1, 1, 1, 1, self.__max_box_per_image, 4))
        warped_image = self.__preprocess(image, warp_points)
        data = (self.__prepare_data(warped_image), dummy_array)
        inference.set_data(data)
        inference.get_meta_dict()['warped_image'] = warped_image
        return inference

    def __out_pipe_process(self, result):
        result, inference = result
        image = inference.get_meta_dict()['warped_image']
        i_dets = RetailInference(image, self.__labels)
        i_dets.decode_netout(result, self.__anchors, self.__nb_class)
        inference.set_result(i_dets)
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        K.set_session(self.__session_runner.get_session())
        self.__tf_sess = K.get_session()
        with self.__tf_sess.as_default():
            with self.__tf_sess.graph.as_default():
                self.__model = load_model(self.__model_path+'/check1.h5',custom_objects={"tf": tf})
                print("Successful load of yolo")


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
                self.__session_runner.get_in_pipe().push(
                    SessionRunnable(self.__job, inference, run_on_thread=self.__run_session_on_thread))
            else:
                self.__in_pipe.wait()

    def __job(self, inference):
        data = inference.get_data()
        self.__out_pipe.push((self.__model.predict([data[0], data[1]])[0], inference))