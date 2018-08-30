import tensorflow as tf
import numpy as np
import cv2
import argparse
# import os
# import re
from flask import Flask, request, Response
from flask_restful import Resource, Api
from flask_cors import CORS


# import json
#from flask.json.jsonify import jsonify

model = None
avg = 0
font = cv2.FONT_HERSHEY_SIMPLEX
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowd Counting')
    parser.add_argument(
        'model',
        type=str,
        help='Path to Model. Model should be on the same path.'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Directory of Model checkpoint folder. Checkpoint should be on the same directory.'
    )
parser.add_argument(
    'video',
    type=str,
    help='Path to the test feed. Video file should be on the same path.'
)
args = parser.parse_args()

model_path = args.model
ckpt_path = args.checkpoint
input_feed = args.video
feed_vid = cv2.VideoCapture(0)
app = Flask(__name__)
CORS(app)
api = Api(app)

success, im = feed_vid.read()


class crowdcount(Resource):
    def get(self):
        global avg
        global success
        global im
        global feed_vid
        avg = 0
        with tf.Session() as sess:
            success = True
            if success:

                print("success-", success)
                new_saver = tf.train.import_meta_graph(model_path)
                new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
                graph = tf.get_default_graph()
                op_to_restore = graph.get_tensor_by_name("add_12:0")
                x = graph.get_tensor_by_name('Placeholder:0')
                fps = feed_vid.get(cv2.CAP_PROP_FPS)
                fps = np.int32(fps)
                print("Frames Per Second:", fps, "\n")
                counter = 0
                avg = 0
                while success:
                    counter += 1
                    img = np.copy(im)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.array(img)
                    img = (img - 127.5) / 128
                    x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
                    x_in = np.float32(x_in)
                    y_pred = []
                    y_pred = sess.run(op_to_restore, feed_dict={x: x_in})
                    sum = np.absolute(np.int32(np.sum(y_pred)))
                    if counter <= fps:
                        avg += sum
                    else:
                        counter = 0
                        avg = np.int32(avg / fps)
                        print(avg)
                        break
                        avg += sum
                    success, im = feed_vid.read()

            cv2.destroyAllWindows()
        return ({"msg": int(avg)}) # Fetches Count


class imageframe(Resource):

    def get_frame(self):
        global im
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()

    def gen(self):
        while True:
            frame = self.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def get(self):
        return Response(self.gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')



api.add_resource(crowdcount, '/crowdcount')  # Route_1
api.add_resource(imageframe, '/feedImg')
if __name__ == '__main__':
     app.run()


