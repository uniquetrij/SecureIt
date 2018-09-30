import cv2

from data.obj_detection.videos import path as videos_path
from obj_detection.yolo_api.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference

# cap = cv2.VideoCapture(-1)
cap = cv2.VideoCapture(videos_path.get() + '/video1.avi')

session_runner = SessionRunner()

frame_no = 0
while True:
    ret, image = cap.read()
    if ret:
        frame_no+=1
        # break
    if frame_no == 125:
        break

detection = YOLOObjectDetectionAPI('yolo_api', True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
detection.use_threading()

session_runner.start()
detection.run()


while True:
    ret, image = cap.read()
    if not ret:
        continue
    detector_ip.push(Inference(image.copy()))
    detector_op.wait()
    ret, inference = detector_op.pull()
    if ret:
        i_dets = inference.get_result()
        # print(i_dets.get_masks()[0].shape)
        frame = i_dets.get_annotated()
        cv2.imshow("", i_dets.get_annotated())
        cv2.waitKey(1)
        # cv2.imwrite("/home/developer/Desktop/folder/" + (str(count).zfill(5)) + ".jpg", frame)
    print(frame_no)
    frame_no+=1
# Thread(target=detect_objects).start()
