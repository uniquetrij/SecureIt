import threading
from os.path import dirname, realpath
from threading import Thread
import tensorflow as tf
from tf_session.tf_session_utils import Pipe


class SessionRunner:
    __config = tf.ConfigProto(log_device_placement=False)
    __config.gpu_options.allow_growth = True
    __counter = 0

    def __init__(self, skip=False):
        self.__self_dir_path = dirname(realpath(__file__))
        self.__thread = None
        self.__pause_resume = None
        self.__tf_sess = tf.Session(config=self.__config)
        self.__in_pipe = Pipe()
        self.__skip = skip

    def get_in_pipe(self):
        return self.__in_pipe

    def get_session(self):
        return self.__tf_sess

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
            self.__in_pipe.pull_wait()
            ret, sess_runnable = self.__in_pipe.pull(self.__skip)
            if ret:
                if type(sess_runnable) is not SessionRunnable:
                    raise Exception("Pipe elements must be a SessionRunnable")
                sess_runnable.execute(self.__tf_sess)


class SessionRunnable:
    def __init__(self, job_fnc, args_dict, run_on_thread=False):
        self.__job_fnc = job_fnc
        self.__args_dict = args_dict
        self.__run_on_thread = run_on_thread

    def execute(self, tf_sess):
        if self.__run_on_thread:
            Thread(target=self.__exec, args=(tf_sess,)).start()
        else:
            self.__exec(tf_sess)

    def __exec(self, tf_sess):
        with tf_sess.as_default():
            with tf_sess.graph.as_default():
                self.__job_fnc(self.__args_dict)