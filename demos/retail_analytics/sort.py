from __future__ import print_function

import json
import time

import numpy as np
import requests

from demos.retail_analytics.data_association import associate_detections_to_trackers
from demos.retail_analytics.kalman_tracker import KalmanBoxTracker
from demos.retail_analytics.util.retail_inference import BoundBox


class Sort:

    def __init__(self,max_age=100,min_hits=10):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = {0:[],1:[],2:[],3:[],4:[],5:[]}

    def update(self, det):
        ret = []

        box_dict = {}
        if det.get_bboxes() is not None:
            bboxes = det.get_bboxes()

            for box in bboxes:
                if box.get_label() in box_dict:
                    box_dict[box.get_label()].append(box.get_bbox())
                else:
                    box_dict[box.get_label()] = [box.get_bbox()]
        for i in self.trackers.keys():
            trks = np.zeros((len(self.trackers[i]), 5))

            to_del = []

            for t, trk in enumerate(trks):
                pos = self.trackers[i][t].predict()
                # print(pos)
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                self.trackers[i].pop(t)

            if det.get_bboxes() is not None:
                det_boxes = []
                if i in box_dict:
                    det_boxes = box_dict[i]


                matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(det_boxes, trks)

                #update matched trackers with assigned detections
                for t,trk in enumerate(self.trackers[i]):
                    if t not in unmatched_trks:
                        d = matched[np.where(matched[:,1]==t)[0],0][0]
                        trk.update(det_boxes[d])
                    # else:
                    #   print(trk.id+1, trk.get_state())

                # create and initialise new trackers for unmatched detections
                for j in unmatched_dets:
                    trk = KalmanBoxTracker(det_boxes[j],i)
                    self.trackers[i].append(trk)

                k = len(self.trackers[i])
                #   print("No. Of People Detected in frame{}".format(i) )

                for trk in reversed(self.trackers[i]):
                    d = trk.get_state()
                    if trk.hit_streak >= self.min_hits:
                        bbox = trk.get_state()
                        label = trk.get_label()
                        box = BoundBox(bbox[0], bbox[1], bbox[2], bbox[3])
                        box.set_label(label)
                        ret.append(box)
                    k -= 1
                    #remove dead tracklet
                    if trk.time_since_update > self.max_age:
                        self.trackers[i].pop(k)


        if len(ret)>0:
            return ret
        return np.empty((0,5))