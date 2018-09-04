from Utils import Pipe
from tf_session.session_runner import SessionRunnable
import numpy as np

PRETRAINED_mars_small128 = 'mars-small128.pb'


class ImageEncoder(SessionRunnable):

    @staticmethod
    def __fetch_model_path(model_name):
        return "../resources/networks/" + model_name

    def __init__(self, model_name=PRETRAINED_mars_small128, graph_prefix=None):

        self.__path_to_frozen_graph = self.__fetch_model_path(model_name)

        if not graph_prefix:
            self.graph_prefix = ''
            graph_prefix = ''
        else:
            self.graph_prefix = graph_prefix + '/'

        self.__in_pipe = Pipe()
        self.__out_pipe = Pipe()

        super(ImageEncoder, self).__init__(graph_prefix, self.__path_to_frozen_graph)

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def run(self, tf_sess, tf_default_graph):
        ret, data_x = self.__in_pipe.pull()
        if not ret:
            return

        if self.__in_pipe.is_closed():
            self.__out_pipe.close()
            return

        self.input_var = tf_default_graph.get_tensor_by_name(
            self.graph_prefix + 'images:0')
        self.output_var = tf_default_graph.get_tensor_by_name(
            self.graph_prefix + 'features:0')

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4

        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        self._run_in_batches(
            lambda x: tf_sess.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size=32)

        self.__out_pipe.push(out)


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
