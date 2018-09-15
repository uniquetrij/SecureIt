import threading
from os.path import dirname, realpath
from threading import Thread
from time import sleep

import tensorflow as tf

from tf_session.tf_session_utils import Pipe


class SessionRunner:
    __config = tf.ConfigProto()
    __config.gpu_options.allow_growth = True
    __counter = 0

    def __init__(self, threading=False):
        self.__self_dir_path = dirname(realpath(__file__))
        self.__thread = None
        self.__pause_resume = None
        self.__sess = tf.Session(config=self.__config)
        self.__in_pipe = Pipe()
        self.__threading = threading

    def get_in_pipe(self):
        return self.__in_pipe

    def get_session(self):
        return self.__sess

    def start(self):
        if self.__thread is None:
            self.__pause_resume = threading.Event()
            self.__thread = Thread(target=self.__start)
            self.__thread.start()

    def stop(self):
        if self.__thread is not None:
            self.__thread = None

    def __start(self):
        while self.__thread:
            ret, pull = self.__in_pipe.pull()
            if ret:
                if self.__threading:
                    Thread(target=self.__exec, args=(pull,)).start()
                else:
                    self.__exec(pull)
            else:
                self.__in_pipe.wait()

    def __exec(self, pull):
        job_fnc, args_dict = pull
        with self.__sess.as_default():
            with self.__sess.graph.as_default():
                job_fnc(args_dict)