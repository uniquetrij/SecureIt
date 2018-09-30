import cv2
from data.obj_tracking.videos import path as videos_path

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture(videos_path.get()+'/Hitman Agent 47 - car chase scene HD.mp4')

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()
session_runner.start()
detection.run()

frame_no = 0
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
        frame = i_dets.get_annotated()
        cv2.imshow("annotated", i_dets.get_annotated())
        # cv2.imshow("annotated", i_dets.extract_patch(0))
        cv2.waitKey(1)
        # person = i_dets.get_category('person')
        # for i in range(i_dets.get_length()):
        #     if i_dets.get_classes(i) == 1 and i_dets.get_scores(i) > 0.7:
        #         cv2.imwrite("/home/uniquetrij/PycharmProjects/SecureIt/data/obj_tracking/outputs/patches/" + (
        #             str(frame_no).zfill(5)) + (str(i).zfill(2)) + ".jpg", i_dets.extract_patches(i))
        print(frame_no)
        frame_no += 1

# Thread(target=detect_objects).start()
