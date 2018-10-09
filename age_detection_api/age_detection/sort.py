from __future__ import print_function

import json
import time

import numpy as np
import requests

from age_detection_api.age_detection.data_association import associate_detections_to_trackers
from age_detection_api.age_detection.kalman_tracker import KalmanBoxTracker


class Sort:

    def __init__(self,max_age=100,min_hits=10):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, det):
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
                face = det.get_crop(bboxes[i])
                trk = KalmanBoxTracker(bboxes[i], ages[i], genders[i], ethnicity[i], face)
                self.trackers.append(trk)
        i = len(self.trackers)
        # print("No. Of People Detected in frame{}".format(i) )
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                ret.append(trk) # +1 as MOT benchmark requires positive
                bbox = trk.get_state()
                # ToDo Handle Face None condition
                # face = det.get_crop(bbox)
                # trk.set_face(face)
            i -= 1
            #remove dead tracklet
            if trk.time_since_update > self.max_age:
                curr = self.trackers.pop(i)
                _visibility = curr.get_visibility()
                _genders = gender = np.average(np.array(curr.get_genders()), axis=0)
                _ages = curr.get_ages()
                url = 'https://us-central1-retailanalytics-d6ccf.cloudfunctions.net/api/persontracking/'
                if ((_visibility) > 15):
                    face = trk.get_face()
                    out_dict = {
                        'serialNo':str(time.time()),
                        'gender': 'Female' if gender[0] > 0.5 else 'Male',
                        'age': int(sum(_ages)/len(_ages))
                    }

                    try:
                        payload = json.dumps(out_dict)
                        print(payload)
                        r = requests.post(url, json = out_dict)
                        print(r)
                    except:
                        print("error")
                    finally:
                        pass



        if len(ret)>0:
            return ret
        return np.empty((0,5))