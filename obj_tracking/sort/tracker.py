import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


class Tracker(object):
    track_count = 0

    def __init__(self, bbox, f_vec):
        self.id = Tracker.track_count
        Tracker.track_count += 1
        self.x = bbox
        self.time_since_update = 0
        self.features = f_vec

    # def convert_bbox_to_z(bbox):
    #     """
    #     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    #       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    #       the aspect ratio
    #     """
    #     w = bbox[2] - bbox[0]
    #     h = bbox[3] - bbox[1]
    #     x = bbox[0] + w / 2.
    #     y = bbox[1] + h / 2.
    #     s = w * h  # scale is just area
    #     r = w / float(h)
    #     return np.array([x, y, s, r]).reshape((4, 1))

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x

    def update(self, bbox, img=None):
        self.time_since_update = 0
        if bbox != []:
            self.x = bbox

    @staticmethod
    def associate_detections_to_trackers(f_vecs, trackers, similarity_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """

        # print("Here...")
        if (len(trackers) == 0):
            print("first time detected")
            return np.empty((0, 2), dtype=int), np.arange(len(f_vecs)), np.empty((0, 4), dtype=int)
        similarity_matrix = np.zeros((len(f_vecs), len(trackers)), dtype=np.float32)

        for d, det in enumerate(f_vecs):
            for t, trk in enumerate(trackers):
                similarity_matrix[d, t] = Tracker.get_cosine_similarity(trk, det)
        '''The linear assignment module tries to minimise the total assignment cost.
        In our case we pass -iou_matrix as we want to maximise the total IOU between track predictions and the frame detection.'''

        matched_indices = linear_assignment(-similarity_matrix)

        # print(type(matched_indices))
        # print(matched_indices)

        unmatched_detections = []
        for d, det in enumerate(f_vecs):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

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
        a = tracker.features
        b = f_vec
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        # print(a.shape)
        # print(b.shape)
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a, b.T)
