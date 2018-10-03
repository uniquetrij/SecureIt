from __future__ import print_function
import numpy as np

from age_detection_api.age_detection.data_association import associate_detections_to_trackers
from age_detection_api.age_detection.kalman_tracker import KalmanBoxTracker


class Sort:

    def __init__(self,max_age=100,min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, det, img=None):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers),5))

        to_del = []
        ret = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()
            #print(pos)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if det.get_bboxes() is not None:
            bboxes = det.get_bboxes()
            ages = det.get_ages()
            genders = det.get_genders()
            ethnicity = det.get_ethnicity()
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(bboxes, trks)

            #update matched trackers with assigned detections
            for t,trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:,1]==t)[0],0][0]
                    trk.update(bboxes[d], ages[d], genders[d], ethnicity[d])
                # else:
                #   print(trk.id+1, trk.get_state())

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                # print(bboxes[i])
                # print(ages[i])
                # print(genders[i])
                # print(ethnicity[i])
                trk = KalmanBoxTracker(bboxes[i], ages[i], genders[i], ethnicity[i])
                self.trackers.append(trk)
        i = len(self.trackers)
        print("No. Of People Detected in frame{}".format(i) )
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret)>0:
            return ret
        return np.empty((0,5))