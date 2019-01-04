import cv2
import time

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from feature_comparator.siamese_api.siamese import SiameseComparator
from data.feature_comparator.siamese_api.trained import path as model_path
from obj_tracking.ofist_api.trail import Trail
from age_detection_api.age_detection.age_api import AgeDetection
from person.person_metadata import Person
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference




class AgeApiRunner(object):

    def __init__(self, session_runner):
        self.__detection = AgeDetection()
        self.__detector_ip = self.__detection.get_in_pipe()
        self.__detector_op = self.__detection.get_out_pipe()
        self.__session_runner = session_runner
        self.__detection.use_session_runner(self.__session_runner)
        self.__detection.use_threading()
        self.__session_runner.start()
        self.__detection.run()

    def get_detector_ip(self):
        return self.__detector_ip
    def get_detector_op(self):
        return self.__detector_op

class Tracker(object):
    num_tracks = 0
    __age_inference = AgeApiRunner(SessionRunner())

    @staticmethod
    def __get_next_id():
        Tracker.num_tracks += 1
        return Tracker.num_tracks

    def __init__(self, bbox, features, patch, frame_no, hit_streak_threshold=10, zones=None):
        self.__zones = zones
        self.__patches = [patch]
        self.__id = self.__get_next_id()
        self.__bbox = bbox
        self.__features_fixed = [features]
        self.__features_update = []
        self.__hit_streak = 0
        self.__time_since_update = 0
        self.__hit_streak_threshold = hit_streak_threshold
        self.__hits = 1
        self.__creation_time = frame_no
        self.__patch_update_timestamp = time.time()
        self.__trail = Trail(self.__zones, self.__id)

        #age detection components
        self.__detect_age = True
        self.__image = None

    def set_image(self, image):
        self.__image = image

    def get_image(self):
        return self.__image

    def detect_age(self):
        return self.__detect_age

    def update_zones(self, zones):
        self.__zones = zones
        if zones is not None:
            self.__trail.update_zones(self.__zones)

    def get_creation_time(self):
        return self.__creation_time

    def get_patches(self):
        return self.__patches

    def get_hits(self):
        return self.__hits

    def get_features(self):
        return self.__features_fixed + self.__features_update

    def get_bbox(self):
        """
        Returns the current bounding box estimate.
        """
        return self.__bbox

    def get_time_since_update(self):
        return self.__time_since_update

    def get_id(self):
        return self.__id

    def is_confident(self):
        return self.__hits > 20

    def update(self, bbox, f_vec, patch):
        timestamp = time.time()
        self.__hits += 1
        if timestamp - self.__patch_update_timestamp > 1:
            if len(self.__features_fixed) < 50:
                self.__features_fixed.append(f_vec)
            self.__patches.append(patch)

            self.__patch_update_timestamp = timestamp
            if len(self.__patches) > 10:
                self.__patches.pop(0)

            if len(self.__features_fixed) > 50:
                self.__features_update.append(f_vec)
                if len(self.__features_update) > 50:
                    self.__features_update.pop(0)

        self.__time_since_update = 0
        self.__hit_streak = min(self.__hit_streak_threshold, self.__hit_streak + 1)

        if bbox:
            self.__bbox = bbox
            self.__trail.update_track(bbox)

    def get_trail(self):
        return self.__trail

    def get_hit_streak(self):
        return self.__hit_streak

    @staticmethod
    def associate_detections_to_trackers(f_vecs, trackers, graph, min_similarity_threshold=0.625):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """

        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(f_vecs)), np.empty((0, 4), dtype=int)

        similarity_matrix = np.zeros((len(f_vecs), len(trackers)), dtype=np.float32)

        for d, det in enumerate(f_vecs):
            for t, trk in enumerate(trackers):
                # x1, y1 = bboxes[d][0], bboxes[d][1]
                # x2, y2 = trk.get_bbox()[0], trk.get_bbox()[1]
                # print((abs(float(y2-y1)**2 - float(x2-x1)**2)**0.5))
                '''100 is probably a very low theshold. Also, We havent done anything for exit frame mechanism. Other than that it seem to work fine'''
                # 100 coz anything lower doesnt seem to detect my walking we might even have to increase it but then worry about its reaction in multi person environment
                # if ((abs(float(y2-y1)**2 - float(x2-x1)**2)**0.5)) < 25:
                #     print((abs(float(y2 - y1) ** 2 - float(x2 - x1) ** 2) ** 0.5))
                #     print(d,t)
                # similarity_matrix[d, t] = Tracker.get_cosine_similarity(trk, det)

                similarity_matrix[d, t] = Tracker.siamese_comparator(trk, det, graph)
        '''The linear assignment module tries to minimise the total assignment cost.
        In our case we pass -iou_matrix as we want to maximise the total IOU between track predictions and the frame detection.'''
        #print("   ----------------matrix")
        #print(similarity_matrix)
        #print("   ----------------")
        matched_indices = linear_assignment(-similarity_matrix)
        #print("matched indices", matched_indices)

        # print("Matched Indices: ", matched_indices[:,1])

        unmatched_detections = []
        for d, det in enumerate(f_vecs):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                # print("Unmatched Tracker: ", t, trackers[t].get_id())
                unmatched_trackers.append(trackers[t].get_id())
                trk.__hit_streak = max(0, trk.__hit_streak - 1)
                trk.__time_since_update += 1

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            trk_index = m[1]
            trk_id = trackers[trk_index].get_id()
            if (similarity_matrix[m[0], m[1]] < min_similarity_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(trk_id)
                # print("Unmatched Tracker ID: ", trk_id)
                trk = trackers[trk_index]
                trk.__hit_streak = max(0, trk.__hit_streak - 1)
                trk.__time_since_update += 1
            else:
                matches.append(np.array([m[0], trk_id]).reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    @staticmethod
    def get_cosine_similarity(tracker, f_vec):
        maximum = 0
        lst = tracker.get_features()
        for a in lst:
            b = f_vec
            a = np.expand_dims(a, axis=0)
            b = np.expand_dims(b, axis=0)
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
            maximum = max(maximum, np.dot(a, b.T)[0][0])
        return maximum

    __siamese_model = None

    @staticmethod
    def siamese_comparator(tracker, f_vec, graph):
        if not Tracker.__siamese_model:
            print("Initializing...")
            with graph.as_default():
                Tracker.__siamese_model = SiameseComparator()()
                Tracker.__siamese_model.load_weights(model_path.get() + '/model_12_28_2018_12_02_56.h5')

        maximum = 0
        lst = tracker.get_features()
        for a in lst:
            b = f_vec
            a = np.expand_dims(a, axis=0)
            b = np.expand_dims(b, axis=0)
            # a = np.expand_dims(a, axis=0)
            # b = np.expand_dims(b, axis=0)
            # a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            # b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
            maximum += Tracker.__siamese_model.predict([a, b])[0][0]
        return maximum / len(lst)

    @staticmethod
    def detect_age_gender(trackers):
        if not Tracker.__age_inference:
            print("initializing Age Model")
            Tracker.__age_inference = AgeApiRunner(SessionRunner())
        detector_ip = Tracker.__age_inference.get_detector_ip()
        detector_op = Tracker.__age_inference.get_detector_op()
        for i, trk in enumerate(trackers):
            trk_trail = trk.get_trail()
            person = trk_trail.get_person()
            if not trk.detect_age() or len(person.get_age_list()) >= 10 or len(person.get_gender_list()) >= 10 or trk.get_image() is None:
                continue
            # print("len of tracker", i, " " ,len(trk.get_patches()))
            detector_ip.push(Inference(trk.get_image().copy()))
            # ret, inference = detector_op.pull(True)

            while True:
                detector_op.wait()
                ret, inference = detector_op.pull(True)
                if ret:
                    # print(ret)
                    if inference.get_result().get_genders() is None or inference.get_result().get_ages() is None:
                        break
                    print(inference.get_result().get_genders())
                    print(inference.get_result().get_ages())
                    gender_confidence = inference.get_result().get_genders()[0][0]
                    gender = 'M' if gender_confidence < 0.5 else 'F'
                    age = int(inference.get_result().get_ages()[0])
                    # print(type())
                    trk.get_trail().get_person().add_age(age)
                    trk.get_trail().get_person().add_gender(gender, gender_confidence)
                    print("ages", trk.get_trail().get_person().get_age_list())
                    print("genders", trk.get_trail().get_person().get_gender_list())
                    break
                # Inference.




