import numpy as np


from obj_tracking.ofist_api.knn_detector import KnnDetector, DistanceMetric




class KNNTracker(object):
    num_tracks = 0

    @staticmethod
    def __get_next_id():
        KNNTracker.num_tracks += 1
        return KNNTracker.num_tracks

    def __init__(self, bbox, features, frame_no, hit_streak_threshold=10):
        self.__id = self.__get_next_id()
        self.__s_id = None
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

    def get_s_id(self):
        return self.__s_id

    def set_s_id(self, s_id):
        self.__s_id = s_id

    def get_hits(self):
        return self.__hits

    def update(self, bbox, f_vec):
        self.__hits+=1
        if len(self.__features_fixed) < 50:
            self.__features_fixed.append(f_vec)
        self.__features_update.append(f_vec)
        if len(self.__features_update) > 50:
            self.__features_update.pop(0)
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
    def associate_detections_to_trackers(f_vecs, trackers, bboxes, distance_threshold=0.375):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """

        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(f_vecs)), np.empty((0, 4), dtype=int)

        # similarity_matrix = np.zeros((len(f_vecs), len(trackers)), dtype=np.float32)

        X, Y = KNNTracker.prepare_data(trackers)
        predictor = KnnDetector()

        matched_indices = []
        unmatched_detections = []
        for i, f_vec in enumerate(f_vecs):
            # id, count, distance = predictor.update(X, Y).observe(f_vec,
            #                                                      distance_metric=DistanceMetric.euclidean_distance).obtain_old(
            #     10, 1)
            # predictor.update(X, Y)
            predictor.update(X, Y)
            eu = predictor.observe(f_vec, distance_metric=DistanceMetric.cosine_distance).obtain(5)
            id1, count1, distance1, _, _ = predictor.get(1)
            try:
                id2, count2, distance2, _, _ = predictor.get(2)
            except:
                pass
            if distance1 < distance_threshold:
                matched_indices.append([i, id1])
                try:
                    print(id1,id2)
                except:
                    pass
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
