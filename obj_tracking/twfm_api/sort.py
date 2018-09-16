"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function

from threading import Thread

import numpy as np

from obj_tracking.twfm_api.image_encoder import ImageEncoder
from obj_tracking.twfm_api.tracker import Tracker
from tf_session.tf_session_utils import Pipe


class Sort:

    def __init__(self, max_age=120, min_hits=3, flush_pipe_on_read=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__outpass_pipe = Pipe(self.__out_pipe_process)

    def __in_pipe_process(self, inference):
        detections = inference.get_boxes_tlbr(normalized=False)
        frame = inference.get_input()
        classes = inference.get_classes()
        person_detections = []
        scores = inference.get_scores()
        for i in range(len(classes)):
            if classes[i] == inference.get_category('person') and scores[i] > .5:
                person_detections.append([detections[i][1], detections[i][0], detections[i][3], detections[i][2]])
        return (frame, person_detections)

    def __out_pipe_process(self, inference):
        image, f_vecs, bboxes = inference

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
            return np.concatenate(ret), image
        return np.empty((0, 5)), image

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self,model, session_runner):
        self.__session_runner = session_runner
        # K.set_session(session_runner.get_session())
        # self.__tf_sess = K.get_session()
        self.__image_encoder = ImageEncoder(model, session_runner)

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            ret, data_tuple = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.data_tuple = data_tuple
                self.__session_runner.add_job(self.__job())
            else:
                self.__in_pipe.wait()

    def __job(self):
        f_vecs = self.__image_encoder.extract_features(self.data_tuple[0], self.data_tuple[1])
        self.__out_pipe.push((self.data_tuple[0], f_vecs, self.data_tuple[1]))




    def update(self, image, bboxes):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """



