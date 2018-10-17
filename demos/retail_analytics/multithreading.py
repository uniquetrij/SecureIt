import csv
from flask_cors import CORS
from flask import request
from flask import Flask, Response
import cv2
from data.demos.retail_analytics.inputs import path as file_path
from tf_session.tf_session_utils import Pipe

app = Flask(__name__)
CORS(app)


def gen_analysis():
    while True:
        stock_in_pipe.wait()
        ret, image = stock_in_pipe.pull()
        if not ret:
            continue
        # print("Pipe pull successful")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')

def gen_tracking():
    while True:
        tracking_in_pipe.wait()
        ret, image = tracking_in_pipe.pull()
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')

def gen_age_api():
    while True:
        age_in_pipe.wait()
        ret, image = age_in_pipe.pull()
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')

stock_in_pipe = Pipe()
tracking_in_pipe = Pipe()
age_in_pipe = Pipe()
point_set_pipe = Pipe()
zone_pipe = Pipe()
dict={'point_set_1':[[103, 13], [551, 14], [535, 341], [114, 343]],'point_set_2':[[99, 11], [552, 13], [542, 339], [108, 342]]}
@app.route('/live_stock_feed')
def live_stock_feed():
    return Response(gen_analysis(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_tracking_feed')
def live_tracking_feed():
    return Response(gen_tracking(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/roi_points',methods=["POST"])
def roi_points():
    global dict
    dict=request.get_json()
    point_set_pipe.push(dict)
    zone_pipe.push(dict)
    with open(file_path.get()+'/demo_zones_1.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['Z1',int(dict['point_set_1'][0][0]),int(dict['point_set_1'][0][1]),
                         int(dict['point_set_1'][1][0]), int(dict['point_set_1'][1][1]),
                         int(dict['point_set_1'][2][0]), int(dict['point_set_1'][2][1]),
                         int(dict['point_set_1'][3][0]), int(dict['point_set_1'][3][1])])
    print("points")
    print(dict)
    return "ok"


@app.route('/live_age_feed')
def live_age_feed():
    return Response(gen_age_api(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run():
    app.run(host='0.0.0.0', debug=True, use_reloader=False,threaded=True)
