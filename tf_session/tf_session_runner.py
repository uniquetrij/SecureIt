from os.path import dirname, realpath
from threading import Thread
import tensorflow as tf


class SessionRunner:

    __config = tf.ConfigProto()
    __config.gpu_options.allow_growth = True

    def __init__(self):
        self.__self_dir_path = dirname(realpath(__file__))
        self.__thread = None
        self.__runnables = []
        self.__sess = tf.Session(config=self.__config)

    def load(self, session_runnable):
        self.__runnables.append(session_runnable)
        if isinstance(session_runnable, SessionRunnable):
            with self.__sess.graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(session_runnable.get_path_to_frozen_graph(), 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name=session_runnable.get_graph_prefix())

        session_runnable.on_load(self.__sess)

    def start(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__start)
            self.__thread.start()

    def stop(self):
        if self.__thread is not None:
            self.__thread = None

    def __start(self):
        with self.__sess.graph.as_default():
            with self.__sess:
                while self.__thread:
                    spawns = []
                    for runnable in self.__runnables:
                        thread = Thread(target=runnable.run())
                        thread.start()
                        spawns.append(thread)


class SessionRunnable:
    def __init__(self, graph_prefix, path_to_frozen_graph):
        self.__name = graph_prefix
        self.__path_to_frozen_graph = path_to_frozen_graph

    def get_graph_prefix(self):
        return self.__name

    def get_path_to_frozen_graph(self):
        return self.__path_to_frozen_graph

    def on_load(self, tf_sess):
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


