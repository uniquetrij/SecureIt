from threading import Thread

import cv2

from data.demos.retail_analytics.inputs import path as input_path
from data.obj_tracking.outputs import path as out_path
from data.videos import path as videos_path
from demos.retail_analytics.ra_person_tracking_api import RAPersonTrackingAPI
from obj_detection.tf_api.tf_object_detection_api import TFObjectDetectionAPI, \
    PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, PRETRAINED_mask_rcnn_inception_v2_coco_2018_01_28
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference
from utils.video_writer import VideoWriter

session_runner = SessionRunner()
session_runner.start()

cap = cv2.VideoCapture(videos_path.get() + '/video1.avi')
# cap = cv2.VideoCapture(-1)
while True:
    ret, image = cap.read()
    if ret:
        break

# detector =  YOLOObjectDetectionAPI('yolo_api', True)
detector = TFObjectDetectionAPI(PRETRAINED_faster_rcnn_inception_v2_coco_2018_01_28, image.shape, 'tf_api', True)
detector.use_session_runner(session_runner)
detector_ip = detector.get_in_pipe()
detector_op = detector.get_out_pipe()
detector.use_threading()
detector.run()

tracker = RAPersonTrackingAPI(input_path.get() + "/zones.csv", flush_pipe_on_read=True, use_detection_mask=False)
tracker.use_session_runner(session_runner)
trk_ip = tracker.get_in_pipe()
trk_op = tracker.get_out_pipe()
tracker.run()


def read():
    count = 0

    while True:
        count += 1
        ret, image = cap.read()
        # if count == 100:
        #     detector_ip.close()
        # print("breaking...")
        # trk_ip.close()
        # break
        if not ret:
            continue
        # image = cv2.resize(image,(image.shape[0]/2,image.shape[1]/2))
        detector_ip.push(Inference(image.copy()))
        # print('waiting')
        detector_op.wait()
        # print('done')
        ret, inference = detector_op.pull()
        if ret:
            i_dets = inference.get_result()
            trk_ip.push(Inference(i_dets))
        # sleep(0.1)


t = Thread(target=read)
t.start()

video_writer = VideoWriter(out_path.get() + "/video1_out.avi", image.shape[1], image.shape[0], 25)

while True:
    # print(detector_op.is_closed())
    trk_op.wait()
    if trk_ip.is_closed():
        # print("Here")
        video_writer.finish()
        break
    ret, inference = trk_op.pull()
    if ret:
        trackers = inference.get_result()
        frame = inference.get_input().get_image()
        patches = inference.get_data()
        # trails = inference.get_meta_dict()['trails']
        for trk in trackers:
            d = trk.get_bbox()
            display = str(int(trk.get_id())) #+ " " + str([z.get_id() for z in trk.get_trail().get_current_zones()])
            l = len(display)
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 1)

            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[0]) + 5 + (10 * l), int(d[1]) + 15), (0, 69, 255),
                          thickness=cv2.FILLED)

            cv2.putText(frame, display, (int(d[0]) + 2, int(d[1]) + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), thickness=1)

            # print(trk.get_id(), [z.get_id() for z in trk.get_trail().get_current_zones()])

            # trail = trails[int(d[4])].get_trail()
            # zone = trail.get_zone()
            # if zone is None and trail :
            #     print()

        overlay = frame.copy()


        # for z in tracker.get_zones():
        #     cv2.polylines(overlay, [np.int32(z.get_coords())], 1,
        #                   (0, 255, 255), 2)

        # frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # # count+=1
        # video_writer.write(frame)

        # if patches:
        #     for i, patch in enumerate(patches):
        #         cv2.imshow("patch" + str(i), patch)
        #         cv2.waitKey(1)
        cv2.imshow("output", frame)
        cv2.waitKey(1)
