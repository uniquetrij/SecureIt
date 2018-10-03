from flask import Flask,  Response, render_template
from age_detection_api import age_detection
import cv2
import PIL


app = Flask(__name__)
age_detect = age_detection.AgeDetection()
video_path = 'input_video/VID_20180822_135103.mp4'

@app.route('/')
def index():
    return render_template('index.html')


def gen_from_video():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        output = cv2.imencode('.jpg', age_detect.pipeline(frame))[1].tobytes()
        # #output = frame
        # print(output.shape)
        # print(type(output))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')


def gen_live():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #age_detect.
        output = age_detect.pipeline(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_from_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
