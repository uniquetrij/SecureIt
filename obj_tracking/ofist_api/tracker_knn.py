import time

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from obj_tracking.ofist_api.knn_detector import KnnDetector


class KNNTracker(object):
    num_tracks = 0

    @staticmethod
    def __get_next_id():
        KNNTracker.num_tracks += 1
        return KNNTracker.num_tracks

    def __init__(self, bbox, features, hit_streak_threshold=10):
        self.__id = self.__get_next_id()
        self.__bbox = bbox
        self.__features_fixed = [features]
        self.__features_update = [features]
        self.__hit_streak = 0
        self.__time_since_update = 0
        self.__hit_streak_threshold = hit_streak_threshold

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

    def update(self, bbox, f_vec):
        if len(self.__features_fixed) < 10:
            self.__features_fixed.append(f_vec)
        # self.__features_update.append(f_vec)
        # if len(self.__features_update) > 50:
        #     self.__features_update.pop(0)
        self.__time_since_update = 0
        self.__hit_streak = min(self.__hit_streak_threshold, self.__hit_streak + 1)

        if bbox:
            self.__bbox = bbox

    def get_hit_streak(self):
        return self.__hit_streak

    @staticmethod
    def prepare_data(trackers):
        X, y = [], []
        for tracker in trackers:
            for f_vec in tracker.get_features():
                X.append(f_vec)
                y.append(tracker.get_id())
        return X, y

    @staticmethod
    def associate_detections_to_trackers(f_vecs, trackers, bboxes, distance_threshold=0.65):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """

        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(f_vecs)), np.empty((0, 4), dtype=int)

        # similarity_matrix = np.zeros((len(f_vecs), len(trackers)), dtype=np.float32)

        X, Y = KNNTracker.prepare_data(trackers)
        predictor = KnnDetector()
        predictor.fit(X, Y)

        matched_indices = []
        unmatched_detections = []
        for i, f_vec in enumerate(f_vecs):
            predictor.observe(f_vec, nearest_count=1)
            id = predictor.get_best_class()
            distance = predictor.get_best_distance()
            if distance < distance_threshold:
                matched_indices.append([i, id])
            else:
                unmatched_detections.append(i)

        matched_indices = np.array(matched_indices)
        unmatched_trackers = []
        for trk in trackers:
            t = trk.get_id()

            if (len(matched_indices) == 0 or t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)
                trk.__hit_streak = max(0, trk.__hit_streak - 1)
                trk.__time_since_update += 1

        return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)

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
            maximum = max(maximum, np.dot(a, b.T))
        return maximum
