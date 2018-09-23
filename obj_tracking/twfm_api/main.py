import time
from threading import Thread
import numpy as np
import cv2
from keras.applications import resnet50

from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28, \
    PRETRAINED_faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28, PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17
from obj_tracking.twfm_api.sort import Sort
from tf_session.tf_session_runner import SessionRunner

# tracker = Sort(max_age=120, min_hits=5)  # create instance of the SORT tracker

model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
model._make_predict_function()
patch = np.random.uniform(0., 255., (224,224,3)).astype(np.uint8)
img = np.expand_dims(patch, axis=0)
img = resnet50.preprocess_input(img)
out = model.observe(img)

session_runner = SessionRunner()
session_runner.start()

cap0 = cv2.VideoCapture(-1)
while True:
    ret, image0 = cap0.read()
    if ret:
        break

detection0 = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image0.shape,
                                  'tf_api_0', True)
detection0.use_session_runner(session_runner)
ip0 = detection0.get_in_pipe()
op0 = detection0.get_out_pipe()
detection0.run()

tracker = Sort(max_age=120, min_hits=3, flush_pipe_on_read=True)
tracker.use_session_runner(model, session_runner)
t_ip = tracker.get_in_pipe()
t_op = tracker.get_out_pipe()
tracker.run()


def readvideo():
    while True:
        re, img0 = cap0.read()
        if re:
            ip0.push(img0.copy())
        time.sleep(0.025)

Thread(target=readvideo).start()



def readinference():
    flag = False
    while True:
        ret, inference = op0.pull()
        if ret:
            t_ip.push(inference)
            while True:
                ret, trackers = t_op.pull()
                if ret:
                    trackers, frame = trackers
                    for d in trackers:
                        cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 0, 0), 2)
                        cv2.putText(frame, str(d[4]), (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                                    (0, 0, 255))

                    cv2.imshow("", frame)
                    cv2.waitKey(1)
                    break
                else:
                    t_op.wait()
        else:
            op0.wait()


Thread(target=readinference).start()





# t = Thread(target=process, args=(op0,"cam0", tracker))
# t.start()
# t.join()


# Thread(target=process, args=(op1,"cam1",)).start()