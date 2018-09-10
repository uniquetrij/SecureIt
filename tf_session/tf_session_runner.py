import threading
from os.path import dirname, realpath
from threading import Thread
from time import sleep

import tensorflow as tf


class SessionRunner:

    __config = tf.ConfigProto()
    __config.gpu_options.allow_growth = True

    def __init__(self):
        self.__self_dir_path = dirname(realpath(__file__))
        self.__thread = None
        self.__pause_resume = None
        self.__runnables = []
        self.__sess = tf.Session(config=self.__config)
        self.__jobs  = []

    def add_job(self, job):
        self.__jobs.append(job)
        self.__pause_resume.set()

    def add(self, session_runnable):
        self.__runnables.append(session_runnable)

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
        with self.__sess.graph.as_default():
            with self.__sess:
                while self.__thread:
                    self.__pause_resume.wait()
                    if self.__jobs:
                        thread = Thread(target=self.__jobs.pop(0))
                        thread.start()
                    else:
                        self.__pause_resume.clear()

class SessionRunnable:
    def __init__(self, graph_prefix, path_to_frozen_graph):
        self.__name = graph_prefix
        self.__path_to_frozen_graph = path_to_frozen_graph

    def get_graph_prefix(self):
        return self.__name

    def get_path_to_frozen_graph(self):
        return self.__path_to_frozen_graph

    def init_graph(self):
        pass

    def run(self):
        pass


class KerasRunnable:
    def __init__(self, graph_prefix):
        self.__name = graph_prefix

    def get_graph_prefix(self):
        return self.__name

    def on_load(self, tf_sess):
        pass

    def run(self):
        pass


