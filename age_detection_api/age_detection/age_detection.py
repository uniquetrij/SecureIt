"""
Face detection
"""
import cv2
import os
import numpy as np
# from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import tensorflow as tf
from random import randint
import json
from time import sleep
import sys
sys.path.insert(0,"/home/developer/agr-master/age_detection_api")
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
race_dict = {0:'White' , 1:'Black' , 2:'Asian' , 3:'Indian' , 4:'Others'}

class AgeDetection(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = r".\pretrained_models\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = r".\pretrained_models\model_new.h5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64, detect_size=25):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AgeDetection, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64, detect_size=25):
        self.face_size = face_size
        self.detect_size = detect_size
        self.model = WideResNet(face_size,  race=True, depth=depth, k=width)()
        self.image = None
        self.image_bounding_boxes = None
        self.face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
        self.save_image_path = 'Extracted'
        self.people_dict = {'total_people':[], 'people_under_age': [], 'people_of_age': []}
        if self.save_image_path not in os.listdir("."):
            os.mkdir(self.save_image_path)
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('model_new.h5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)
        self.graph = tf.get_default_graph()



    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

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

    def detect_face(self):

        underage = 0
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.16,
#             minNeighbors=5,
#             minSize=(25, 25)
#         )
        faces = detector.detect_faces(self.image)
        self.people_dict['total_people'].append(len(faces))
        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(self.image, face, margin=40, size=self.face_size)
            (x, y, w, h) = cropped
            #face_img = (face_img/255)-0.5
            cv2.rectangle(self.image_bounding_boxes, (x, y), (x + w, y + h), (255, 200, 0), 2)
            face_imgs[i,:,:,:] = face_img
        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            with self.graph.as_default():
                results = self.model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            predicted_race = np.argmax(results[2] , axis=1)

        # draw results
        
        for i, face in enumerate(faces):
            label = "{},{},{}".format(int(predicted_ages[i]),
                                    "F" if predicted_genders[i][0] > 0.5 else "M" , race_dict[predicted_race[i]])
            if int(predicted_ages[i]) < 18:
                underage += 1
            self.draw_label(self.image_bounding_boxes, (face[0], face[1]), label)
        self.people_dict['people_under_age'].append(underage)
        self.people_dict['people_of_age'].append(len(faces) - underage)

    def read_image(self, image_path):
        self.image = cv2.imread(image_path)

    def set_image(self, image):
        self.image = image
        self.image_bounding_boxes = image

    def save_image(self, image):
        name = str(randint(0, 10000)) + ".jpg"
        img_path = os.path.join(self.save_image_path, name)
        cv2.imwrite(img_path, image)

    def create_metadata(self):
        with open('data_disp.json', 'w') as fp:
            json.dump(self.people_dict, fp)

    def pipeline(self, img):
        # face_detect = FaceDetection(save_image_flag = False)
        self.set_image(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
        self.detect_face()
        return self.image_bounding_boxes

    def live_cv2(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            if not video_capture.isOpened():
                sleep(5)
            ret, image = video_capture.read()
            self.set_image(image)
            self.detect_face()
            cv2.imshow('age_detection_api',self.image_bounding_boxes)
            if cv2.waitKey(5) == 27:
                break

        video_capture.release()
        cv2.distroyAllWindows()

    def live_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, image = cap.read()
            self.set_image(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
            self.detect_face()
            cv2.imshow('age_detection_api detection',self.image_bounding_boxes)
            if cv2.waitKey(5) == 27:
                break
        cap.release()
        cv2.distroyAllWindows()


