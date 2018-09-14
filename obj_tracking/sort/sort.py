"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function

import numpy as np

from obj_tracking.sort.correlation_tracker import CorrelationTracker
from obj_tracking.sort.image_encoder import ImageEncoder


# from kalman_tracker import KalmanBoxTracker
# from correlation_tracker import CorrelationTracker
# from data_association import associate_detections_to_trackers
from obj_tracking.sort.tracker import Tracker


class Sort:

    def __init__(self, max_age=1, min_hits=3, use_dlib=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.__image_encoder = ImageEncoder()
        self.use_dlib = use_dlib

    def update(self, bboxes, img):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # print(type(dets))

        self.frame_count += 1
        # get predicted locations from existing trackers.
        f_vecs = self.__image_encoder.get_features(img, bboxes)

        print(f_vecs.shape)
        # print(f_vecs)
        # if not self.trackers:
        #     for i in range(len(bboxes)):
        #         self.trackers.append(Tracker(bboxes[i], f_vecs[i]))
        # else:
        #     Tracker.associate_detections_to_trackers(f_vecs, self.trackers)

        if bboxes != []:
            print("Here...")
            matched, unmatched_dets, unmatched_trks = Tracker.associate_detections_to_trackers(f_vecs, self.trackers)
            print("matched", len(matched))
            print("unmatched", len(unmatched_dets))
            print("trks", len(unmatched_trks))

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if (t not in unmatched_trks):
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    print(d)
                    trk.update(bboxes[d[0]], f_vecs[d[0]])  ## for dlib re-intialize the trackers ?!

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                if not self.use_dlib:
                    trk = Tracker(bboxes[i], f_vecs[i])
                else:
                    trk = CorrelationTracker(bboxes[i], img)
                self.trackers.append(trk)

        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            # if bboxes == []:
                # trk.update([], img)
            d = trk.get_state()
            # if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            # if (trk.time_since_update > self.max_age):
            #     self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

