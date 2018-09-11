from threading import Thread
import tensorflow as tf
from tf_session.tf_session_utils import Pipe
from tf_session.tf_session_runner import SessionRunnable
import numpy as np

PRETRAINED_mars_small128 = 'mars-small128.pb'


class ImageEncoder:

    @staticmethod
    def __fetch_model_path(model_name):
        return "../resources/networks/" + model_name

    def __init__(self, session_runner, model_name=PRETRAINED_mars_small128, graph_prefix=None):
        self.__path_to_frozen_graph = self.__fetch_model_path(model_name)

        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'


        self.__session_runner = session_runner
        self.__tf_sess = session_runner.get_session()
        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        self.init_graph()

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def __in_pipe_process(self, obj):
        return obj

    def __out_pipe_process(self, obj):
       return obj

    def init_graph(self):

        with self.__tf_sess.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.__path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name=self.__graph_prefix)

        self.input_var = self.__tf_sess.graph.get_tensor_by_name(
            self.__graph_prefix + 'images:0')
        self.output_var = self.__tf_sess.graph.get_tensor_by_name(
            self.__graph_prefix + 'features:0')

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4

        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()


    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            ret, data_x = self.__in_pipe.pull()
            if ret:
                self.data_x = data_x
                self.__session_runner.add_job(self.__job())
            else:
                self.__in_pipe.wait()

    def __job(self):
        __tracking_features = np.zeros((len(self.data_x), self.feature_dim), np.float32)
        self._run_in_batches(
            lambda x: self.__tf_sess.run(self.output_var, feed_dict=x),
            {self.input_var: self.data_x}, __tracking_features, batch_size=32)
        self.__out_pipe.push(__tracking_features)

    def _run_in_batches(self, f, data_dict, out, batch_size):
        data_len = len(out)
        num_batches = int(data_len / batch_size)

        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
            out[s:e] = f(batch_data_dict)
        if e < len(out):
            batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
            out[e:] = f(batch_data_dict)
