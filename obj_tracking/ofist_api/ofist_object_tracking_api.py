import random
from threading import Thread

import cv2

from feature_extraction.rn50_api.resnet50_api import ResNet50ExtractorAPI
from feature_extraction.mars_api.mars_api import MarsExtractorAPI
from obj_tracking.ofist_api import retinex, enhancer
from obj_tracking.ofist_api.enhancer import ImageEnhancer
from obj_tracking.ofist_api.tracker import Tracker
from obj_tracking.ofist_api.tracker_knn import KNNTracker
from obj_tracking.ofist_api.zone import Zone
from tf_session.tf_session_utils import Pipe, Inference
import numpy as np


class OFISTObjectTrackingAPI:

    def __init__(self, max_age=10000, min_hits=5, flush_pipe_on_read=False, use_detection_mask=False, conf_path=None):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.__bg_frame = None
        self.__bg_gray = None
        self.__conf_path = conf_path
        self.__flush_pipe_on_read = flush_pipe_on_read

        # self.__feature_dim = (128)
        # self.__image_shape = (128, 64, 3)

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        self.__use_detection_mask = use_detection_mask
        self.__zones = None
        if self.__conf_path is not None:
            self.__zones = Zone.create_zones_from_conf(self.__conf_path)

    number = 0

    def __extract_image_patch(self, image, bbox, patch_shape):

        sx, sy, ex, ey = np.array(bbox).astype(np.int)

        dx = ex - sx
        dy = ey - sy

        mx = int((sx + ex) / 2)
        my = int((sy + ey) / 2)

        dx = int(min(40, dx / 2))
        dy = int(min(50, dy / 2))
        # image = image[sy:my + dy, mx - dx:mx + dx]
        # image = ImageEnhancer.gaussian_blurr(image, sigma=1.75)
        # image = ImageEnhancer.lab_enhancement(image, l=0.75)
        # image = ImageEnhancer.hsv_enhancement(image, s=10, v=5)
        # image = ImageEnhancer.hls_enhancement(image, l=2)
        # image = ImageEnhancer.lab_enhancement(image, l=1.25)
        # image = ImageEnhancer.gamma_correction(image, gamma=3)
        # image = ImageEnhancer.gaussian_blurr(image, sigma=1.1)

        # dx = int(min(60, dx / 2))
        # dy = int(min(90, dy / 2))
        image = image[sy:my + dy, mx - dx:mx + dx]
        image = ImageEnhancer.gaussian_blurr(image, sigma=1.75)
        image = ImageEnhancer.lab_enhancement(image, l=0.125)
        image = ImageEnhancer.hsv_enhancement(image, s=5, v=5)
        image = ImageEnhancer.hls_enhancement(image, l=2)
        # image = ImageEnhancer.lab_enhancement(image, l=1)
        # image = ImageEnhancer.gamma_correction(image, gamma=3)
        image = ImageEnhancer.gaussian_blurr(image, sigma=1.25)

        # image = image[sy:ey, sx:ex]


        # image = ImageEnhancer.gaussian_blurr(image, sigma=2)
        # image = ImageEnhancer.lab_enhancement(image, l=0.75)
        # image = ImageEnhancer.hsv_enhancement(image, s=3, v=2)
        # image = ImageEnhancer.lab_enhancement(image, l=1.25)
        # image = ImageEnhancer.gamma_correction(image, gamma=3)

        # image = ImageEnhancer.preprocess_retinex(image)

        image = cv2.resize(image, tuple(patch_shape[::-1]))

        return image

    def __in_pipe_process(self, inference):
        i_dets = inference.get_input()
        frame = i_dets.get_image()
        classes = i_dets.get_classes()
        boxes = i_dets.get_boxes_tlbr(normalized=False)
        masks = i_dets.get_masks()
        bboxes = []

        scores = i_dets.get_scores()
        for i in range(len(classes)):
            if classes[i] == i_dets.get_category('person') and scores[i] > .95:
                bboxes.append([boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]])
        patches = [0 for x in bboxes]
        # flips = [0 for x in bboxes]
        threads = []

        for i in range(len(bboxes)):
            box = bboxes[i]
            if self.__use_detection_mask:
                mask = masks[i]
                mask = np.stack((mask, mask, mask), axis=2)
                image = np.multiply(frame, mask)
            else:
                image = frame

            def exec(patches, index):
                index = i
                patch = self.__extract_image_patch(image, box, self.__image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(box))
                    patch = np.random.uniform(0., 255., self.__image_shape).astype(np.uint8)
                if bool(random.getrandbits(1)):
                    patches[index] = patch
                else:
                    patches[index] = cv2.flip(patch,1)

            threads.append(Thread(target=exec, args=(patches,i, )))
            threads[-1].start()

        for thread in threads:
            thread.join()

        inference.set_data(patches)
        inference.get_meta_dict()['bboxes'] = bboxes
        return inference

    def __out_pipe_process(self, inference):
        f_vecs = inference.get_result()
        inference = inference.get_meta_dict()['inference']
        bboxes = inference.get_meta_dict()['bboxes']
        patches = inference.get_data()
        self.frame_count += 1

        matched, unmatched_dets, unmatched_trks = Tracker.associate_detections_to_trackers(f_vecs, self.trackers,
                                                                                           bboxes)

        if bboxes:
            for trk in self.trackers:
                if (trk.get_id() not in unmatched_trks):
                    d = matched[np.where(matched[:, 1] == trk.get_id())[0], 0][0]
                    trk.update(bboxes[d], f_vecs[d], patches[d])

            for i in unmatched_dets:
                trk = Tracker(bboxes[i], f_vecs[i], patches[i], self.frame_count, zones=self.__zones)
                self.trackers.append(trk)

        i = len(self.trackers)
        ret = []
        trails = {}
        for trk in reversed(self.trackers):
            if (trk.get_hit_streak() >= self.min_hits):  # or self.frame_count <= self.min_hits):
                ret.append(trk)
            i -= 1

            if (trk.get_time_since_update() > self.max_age):
                self.trackers.pop(i)
            if self.frame_count - trk.get_creation_time() >= 30 and not trk.is_confident():
                self.trackers.pop(i)
            trails[trk.get_id()] = trk.get_trail()
            #
        inference.get_meta_dict()['trails'] = trails

        if (len(ret) > 0):
            inference.set_result(ret)
        else:
            inference.set_result(np.empty((0, 5)))
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def get_zones(self):
        return self.__zones

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        # self.__encoder = ResNet50ExtractorAPI("", True)
        self.__encoder = MarsExtractorAPI(flush_pipe_on_read=True)
        self.__encoder.use_session_runner(session_runner)
        self.__image_shape = self.__encoder.get_input_shape()
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
                self.__enc_in_pipe.close()
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
