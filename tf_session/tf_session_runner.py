import threading
from os.path import dirname, realpath
from threading import Thread
from time import sleep

import tensorflow as tf


class SessionRunner:
    __config = tf.ConfigProto()
    __config.gpu_options.allow_growth = True
    __counter = 0

    def __init__(self, is_live=True):
        self.__self_dir_path = dirname(realpath(__file__))
        self.__thread = None
        self.__pause_resume = None
        self.__runnables = []
        self.__sess = tf.Session(config=self.__config)
        self.__jobs = []
        self.__args = []
        self.__live = is_live

    def add_job(self, job, args):
        self.__jobs.append(job)
        self.__args.append(args)
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
        while self.__thread:
            self.__pause_resume.wait()
            if self.__live:
                self.__exec()
            else:
                Thread(target=self.__exec).start()



    def __exec(self):
        with self.__sess.as_default():
            with self.__sess.graph.as_default():
                if self.__jobs:
                    try:
                        self.__jobs.pop(0)(self.__args.pop(0))
                    except:
                        pass
                else:
                    self.__pause_resume.clear()