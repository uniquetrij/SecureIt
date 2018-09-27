import time

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from feature_comparator.siamese_api.siamese import SiameseComparator
from data.feature_comparator.siamese_api.trained import path as model_path

class Tracker(object):

    num_tracks = 0

    @staticmethod
    def __get_next_id():
        Tracker.num_tracks += 1
        return Tracker.num_tracks

    def __init__(self, bbox, features, frame_no, hit_streak_threshold = 10):
        self.__id = self.__get_next_id()
        self.__bbox = bbox
        self.__features_fixed = [features]
        self.__features_update = [features]
        self.__hit_streak = 0
        self.__time_since_update = 0
        self.__hit_streak_threshold = hit_streak_threshold
        self.__hits = 1
        self.__creation_time = frame_no

    def get_creation_time(self):
        return self.__creation_time

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

    def update(self, bbox, f_vec):
        self.__hits += 1
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
    def associate_detections_to_trackers(f_vecs, trackers, bboxes, graph, similarity_threshold=0.3):
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
                #similarity_matrix[d, t] = Tracker.get_cosine_similarity(trk, det)
                similarity_matrix[d, t] = Tracker.siamese_comparator(trk, det, graph)
        '''The linear assignment module tries to minimise the total assignment cost.
        In our case we pass -iou_matrix as we want to maximise the total IOU between track predictions and the frame detection.'''

        matched_indices = linear_assignment(-similarity_matrix)

        unmatched_detections = []
        for d, det in enumerate(f_vecs):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)
                trk.__hit_streak = max(0, trk.__hit_streak-1)
                trk.__time_since_update += 1

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (similarity_matrix[m[0], m[1]] < similarity_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:

                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    @staticmethod
    def get_cosine_similarity(tracker, f_vec):
        maximum = 0
        lst = tracker.get_features()
        for a in  lst:
            b = f_vec
            a = np.expand_dims(a, axis=0)
            b = np.expand_dims(b, axis=0)
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
            maximum = max(maximum, np.dot(a, b.T)[0][0])
        return maximum

    __siamese_model = None
    @staticmethod
    def siamese_comparator( tracker, f_vec, graph):
        if not  Tracker.__siamese_model:
            print("Initializing...")
            with graph.as_default():
                Tracker.__siamese_model = SiameseComparator()()
                Tracker.__siamese_model.load_weights(model_path.get() + '/siamese-mars-small128.h5')

        maximum = 0
        lst = tracker.get_features()
        for a in lst:
            b = f_vec
            a = np.expand_dims(a, axis=0)
            b = np.expand_dims(b, axis=0)
            a = np.expand_dims(a, axis=0)
            b = np.expand_dims(b, axis=0)
            # a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            # b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
            maximum += Tracker.__siamese_model.predict([a, b])[0][0]
        return maximum/len(lst)
