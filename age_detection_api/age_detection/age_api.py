from threading import Thread
from age_detection_api.utils.age_inference import AgeInference
from age_detection_api.wide_resnet import WideResNet
from mtcnn.mtcnn import MTCNN
from tf_session.tf_session_runner import SessionRunnable
from tf_session.tf_session_utils import Pipe
from keras import backend as K
import numpy as np
import cv2
from data.age_detection.trained import path as age_model_path
class AgeDetection(object):

    '''
    Path to the trained model path for age_detection
    '''
    WRN_WEIGHTS_PATH = age_model_path.get() + '/age_model.h5'

    def __init__(self, flush_pipe_on_read=False):
        self.__detector = None
        self.__face_size = 64
        self.__flush_pipe_on_read = flush_pipe_on_read
        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def __in_pipe_process(self, inference):
        '''
        :param inference: inference object with input frame
        :return: inference: inference object with age_inference object containing face bbox details,
                data condaining np_array of all faces detected
        '''
        frame = inference.get_input()
        faces = self.__detector.detect_faces(frame)
        faces = list_filter(faces, 0.90)
        # print("faces " ,faces)
        bboxes = []
        face_imgs = np.empty((len(faces), self.__face_size, self.__face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(frame, face[:4], margin=40, size=self.__face_size)
            (x, y, w, h) = cropped
            bboxes.append([x, y, x + w, y + h])
            # face_img = (face_img/255)-0.5
            # cv2.rectangle(self.image_bounding_boxes, (x, y), (x + w, y + h), (255, 200, 0), 2)
            face_imgs[i, :, :, :] = face_img
        age_inference = AgeInference(frame, bboxes=bboxes)
        inference.set_result(age_inference)
        inference.set_data(face_imgs)
        return inference

    def __out_pipe_process(self, inference):
        '''
        :param inference: inference object with faces np array in data,
                age_inference object with bboxes in result
        :return: inference: inference object with age_inference object in result after processing
                    face np array from data for age, gender and ethnicity
        '''
        # placeholder for cropped faces
        results, inference = inference
        age_inference = inference.get_result()
        if results is not None:
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            predicted_ethnicity = np.argmax(results[2], axis=1)
            age_inference.set_ages(predicted_ages)
            age_inference.set_genders(predicted_genders)
            age_inference.set_ethnicity(predicted_ethnicity)
        inference.set_result(age_inference)
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_threading(self, run_on_thread=True):
        self.__run_session_on_thread = run_on_thread

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        K.set_session(self.__session_runner.get_session())
        self.__tf_sess = K.get_session()
        self.__detector = MTCNN(session_runner=self.__session_runner)

        with self.__tf_sess.as_default():
            self.__model = WideResNet(self.__face_size)()
            self.__model.load_weights(AgeDetection.WRN_WEIGHTS_PATH)

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
        '''
        :param inference: run the model on data from inference object and push it to out_pipe
        :return:
        '''
        try:
            self.__out_pipe.push((self.__model.predict(inference.get_data()), inference))
        except:
            self.__out_pipe.push((None, inference))


def list_filter(lst, confidence):
    out = []
    for l in lst:
        if l[4] < confidence:
            continue
        out.append(l)
    return out