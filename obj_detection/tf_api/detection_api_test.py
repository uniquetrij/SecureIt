import cv2

from obj_detection.tf_api.detection_api import TFObjectDetectionAPI, PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28
from tf_session.session_runner import SessionRunner

if __name__ == '__main__':
    tfSession = SessionRunner()
    detection = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, 'tf_api')
    ip = detection.get_in_pipe()
    op = detection.get_out_pipe()
    tfSession.load(detection)

    tfSession.start()

    cap = cv2.VideoCapture(-1)
    for i in range(1000):
        ret, image = cap.read()
        if not ret:
            continue
        ip.push(image.copy())

        ret, inference = op.pull()
        if ret:
            cv2.imshow("", inference.get_annotated())
            # print(inference.classes[0])
            cv2.waitKey(1)

    tfSession.stop()
