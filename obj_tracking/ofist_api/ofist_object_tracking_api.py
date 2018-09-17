from threading import Thread

import cv2

from feature_extraction.rn50_api.resnet50_api import ResNet50ExtractorAPI
from obj_tracking.ofist_api.tracker import Tracker
from tf_session.tf_session_utils import Pipe, Inference
import numpy as np


class OFISTObjectTrackingAPI:

    def __init__(self, max_age=10000, min_hits=5, flush_pipe_on_read=False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__feature_dim = (2048)
        self.__image_shape = (224, 224, 3)

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

    def __extract_image_patch(self, image, bbox, patch_shape):
        sx, sy, ex, ey = np.array(bbox).astype(np.int)
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    def __in_pipe_process(self, inference):
        i_dets = inference.get_input()
        frame = i_dets.get_image()
        classes = i_dets.get_classes()
        boxes = i_dets.get_boxes_tlbr(normalized=False)
        bboxes = []
        scores = i_dets.get_scores()
        for i in range(len(classes)):
            if classes[i] == i_dets.get_category('person') and scores[i] > .5:
                bboxes.append([boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]])
        patches = []
        for box in bboxes:
            patch = self.__extract_image_patch(frame, box, self.__image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., self.__image_shape).astype(np.uint8)
            patches.append(patch)

        inference.set_data(patches)
        inference.get_meta_dict()['bboxes'] = bboxes
        return inference

    def __out_pipe_process(self, inference):
        f_vecs = inference.get_result()
        inference = inference.get_meta_dict()['inference']
        bboxes = inference.get_meta_dict()['bboxes']
        self.frame_count = min(self.frame_count + 1, 1000)

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
            d = trk.get_bbox()
            if (trk.get_hit_streak() >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.get_id()])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.get_time_since_update() > self.max_age):
                self.trackers.pop(i)

        if (len(ret) > 0):
            inference.set_result(np.concatenate(ret))
        else:
            inference.set_result(np.empty((0, 5)))
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__encoder = ResNet50ExtractorAPI(flush_pipe_on_read=True)
        self.__encoder.use_session_runner(session_runner)
        self.__enc_in_pipe = self.__encoder.get_in_pipe()
        self.__enc_out_pipe = self.__encoder.get_out_pipe()
        self.__encoder.run()

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__job(inference)
            else:
                self.__in_pipe.wait()

    def __job(self, inference):
        self.__enc_in_pipe.push(
            Inference(inference.get_data(), meta_dict={'inference': inference}, return_pipe=self.__out_pipe))
