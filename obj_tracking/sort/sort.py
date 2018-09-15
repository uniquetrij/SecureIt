"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function

import time

import numpy as np
from obj_tracking.sort.image_encoder import ImageEncoder
from obj_tracking.sort.tracker import Tracker


class Sort:

    def __init__(self, max_age=120, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.__image_encoder = ImageEncoder()

    def update(self, image, bboxes):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        timestamp = time.time()
        self.frame_count += 1
        # get predicted locations from existing trackers.
        f_vecs = self.__image_encoder.extract_features(image, bboxes)

        if bboxes:
            matched, unmatched_dets, unmatched_trks = Tracker.associate_detections_to_trackers(f_vecs, self.trackers)


            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if (t not in unmatched_trks):
                    d = matched[np.where(matched[:, 1] == t)[0], 0][0]
                    trk.update(bboxes[d], f_vecs[d])  ## for dlib re-intialize the trackers ?!

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = Tracker(bboxes[i], f_vecs[i])
                self.trackers.append(trk)

        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            # if bboxes == []:
                # trk.update([], img)
            d = trk.get_bbox()
            if (trk.get_hit_streak() >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.get_id()])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.get_time_since_update() > self.max_age):
                self.trackers.pop(i)
        print(len(self.trackers))
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

