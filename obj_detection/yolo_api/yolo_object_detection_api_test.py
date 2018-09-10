import cv2

from obj_detection.yolo_api.yolo_keras.yolo_keras_object_detection_api import YOLOObjectDetectionAPI
from tf_session.tf_session_runner import SessionRunner

if __name__ == '__main__':

    tfSession = SessionRunner()

    cap = cv2.VideoCapture(-1)

    detection = YOLOObjectDetectionAPI('yolo_api')
    ip = detection.get_in_pipe()
    op = detection.get_out_pipe()
    tfSession.load(detection)

    tfSession.start()

    # detection = YOLO()
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())
