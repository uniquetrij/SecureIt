from __future__ import print_function
import numpy as np

from tailgating_detection.data_association import associate_detections_to_trackers
from tailgating_detection.kalman_tracker import KalmanBoxTracker



class Sort:

  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

    self.x = np.array([1720, 1520])
    self.y = np.array([0, 1080])
    self.line = np.polyfit(self.x, self.y, 1)
    self.line_func = np.poly1d(self.line)
    self.in_counter = 0
    self.out_counter = 0

  def update(self,dets,img=None):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    # print(len(trks))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].observe(img) #for kal!
      #print(pos)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    if dets != []:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
          d = matched[np.where(matched[:,1]==t)[0],0]
          trk.update(dets[d,:][0],img) ## for dlib re-intialize the trackers ?!
        # else:
        #   print(trk.id+1, trk.get_state())

      # create and initialise new trackers for unmatched detections
      for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], self.line_func)
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        if dets == []:
          trk.update([],img)
        d = trk.get_state()
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          x1 = int(d[0])
          y1 = int(d[1])
          x2 = int(d[2])
          y2 = int(d[3])
          w = x2 - x1
          h = y2 - y1
          cX = int(x1 + w/2)
          cY = int(y1 + h/2)
          if (trk.flag == 0) and ((cY - self.line_func(cX)) < 0):
            self.in_counter+=1
            trk.flag = 1
          if (trk.flag == 1) and ((cY - self.line_func(cX)) > 0):
            self.out_counter += 1
            trk.flag = 0
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))