from threading import Thread

import cv2

from feature_extraction.resnet50_api import ResNet50ExtractorAPI
from tf_session.tf_session_utils import Pipe
import numpy as np


class OFISTObjectTrackingAPI:

    def __init__(self, max_age=120, min_hits=3, flush_pipe_on_read=False):
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
        boxes = inference.get_boxes_tlbr(normalized=False)
        frame = inference.get_image()
        classes = inference.get_classes()
        bboxes = []
        scores = inference.get_scores()
        for i in range(len(classes)):
            if classes[i] == inference.get_category('person') and scores[i] > .5:
                bboxes.append([boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]])
        patches = []
        for box in bboxes:
            patch = self.__extract_image_patch(inference.get_image(), box, self.__image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., self.__image_shape).astype(np.uint8)
            patches.append(patch)
        return (frame, patches)

    def __out_pipe_process(self, inference):
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__encoder = ResNet50ExtractorAPI()
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

            ret, data = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__enc_in_pipe.push(data[1])
            else:
                self.__in_pipe.wait()

    # def __out(self):
        # while self.__thread:
        #
        #     if self.__in_pipe.is_closed():
        #         self.__out_pipe.close()
        #         return
        #
        #     ret, f_vecs = self.__enc_out_pipe.pull()
        #     if ret:
        #         self.__out_pipe.push(())
        #     else:
        #         self.__in_pipe.wait()
