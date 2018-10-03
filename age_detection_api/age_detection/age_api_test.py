import time

import cv2

from age_detection_api.age_detection.age_api import AgeDetection
from age_detection_api.age_detection.sort import Sort
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture(videos_path.get()+'/Hitman Agent 47 - car chase scene HD.mp4')

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = AgeDetection()
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()
session_runner.start()
detection.run()
tracker = Sort()


frame_no = 0
try:
    start = time.time()
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        detector_ip.push(Inference(image.copy()))
        detector_op.wait()
        ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            # print(i_dets.get_masks()[0].shape)
            frame = i_dets.get_image()
            # person = i_dets.get_category('person')
            # for i in range(i_dets.get_length()):
            #     if i_dets.get_classes(i) == 1 and i_dets.get_scores(i) > 0.7:
            #         cv2.imwrite("/home/uniquetrij/PycharmProjects/SecureIt/data/obj_tracking/outputs/patches/" + (
            #             str(frame_no).zfill(5)) + (str(i).zfill(2)) + ".jpg", i_dets.extract_patches(i))
            #print(frame_no)
            trackers = tracker.update(i_dets)
            for trk in trackers:
                bbox = trk.get_state()
                ages = trk.get_ages()
                genders = trk.get_genders()
                ethnicity = trk.get_ethnicity()
                age = sum(ages) / len(ages)
                # gender = np.sum()
                frame = i_dets.annotate(frame, bbox, int(age), genders[-1], ethnicity[-1])

            frame_no += 1

            cv2.imshow("Final Output", frame)
            cv2.waitKey(1)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    end = time.time()
    print(frame_no / (end - start))