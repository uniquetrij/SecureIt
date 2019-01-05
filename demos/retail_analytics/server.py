import jsonpickle as jsonpickle
from flask import Flask, render_template, Response
import cv2
import os.path
from flask_cors import CORS
from flask import request
from flask import jsonify
import numpy as np
import matplotlib.pyplot as plt

# from tf_api.person_item_association import Person_Item_Association
from tf_session.tf_session_utils import Pipe

app = Flask(__name__)
CORS(app)

# person_association = Person_Item_Association()
processed_zone_camera_feed={}
stock_in_pipe = Pipe()

@app.route('/hello')
def hello():
    region = request.args.get('region',None)
    pipe = request.args.get('pipe',None)
    return jsonify({'region':region,'pipe':pipe})

@app.route('/notify_zone_entry', methods=['POST'])
def zone_entry():

    content = request.files('image')
    zone_id = content['zone_id']
    person_id = content['person_id']
    # image=content['image_str']
    # print(zone_id,person_id)
    print("Zone "+str(zone_id)+" Person : "+str(person_id))

    #testing only

    person_association.start_zone_camera_analysis(person_id,zone_id)

    return jsonify("person tracking and cart analysis started")

@app.route('/notify_zone_exit', methods=['POST'])
def zone_exit():
    content = request.get_json()
    zone_id = content['zone_id']
    person_id = content['person_id']
    print("EXITING: "+zone_id + " " + str(person_id))
    # person_association.stop_zone_camera_analysis()

    return Response("person tracking and analysis stopped..")

@app.route('/store_processed_zone_camera_feed', methods=['POST'])
def store_processed_zone_camera_feed():
    content = request.get_json()
    camera_id = content['camera_id']
    frame = content['frame']
    print("camera_id")
    global processed_zone_camera_feed
    processed_zone_camera_feed[camera_id] = frame

    return jsonify("success")

@app.route('/get_processed_zone_camera_feed')
def get_processed_zone_camera_feed():

    camera_id = request.args.get('camera_id',None)

    global processed_zone_camera_feed
    frame = get_processed_zone_camera_feed[camera_id]

    return jsonify("success")

# @app.route('/get_cart_details')
# def get_cart_details():
#
#     #testing
#     # cart = person_association.get_cart_details()
#     return jsonify(str(cart))


# @app.route('/process_video', methods=['POST'])
# def postJsonHandler():
#     # print (request.is_json)
#     global region_selected
#     global pipe_selected
#
#     content = request.get_json()
#     # print (content)
#     region_selected = content['region']
#     pipe_selected = content['pipe']
#     print(region_selected, pipe_selected)
#     inputUrl = region_selected + pipe_selected
#     if not os.path.exists('../videos/output/' + region_selected + pipe_selected + '_output.mp4'):
#         processVideo(inputUrl)
#     return jsonify('Ok')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_zone_camera_feed')
def input_video_feed():
    global region_selected
    global pipe_selected
    return Response(open('../videos/input/' + region_selected + pipe_selected, "rb"), mimetype="video/mp4")


@app.route('/output_video_feed')
def output_video_feed():
    global region_selected
    global pipe_selected
    return Response(open('../videos/output/' + region_selected + pipe_selected + '_output.mp4', "rb"),
                    mimetype="video/mp4")

# @app.route('/process_video', methods=['POST'])
# def postJsonHandler():
#     # print (request.is_json)
#     global region_selected
#     global pipe_selected
#
#     content = request.get_json()
#     # print (content)
#     region_selected = content['region']
#     pipe_selected = content['pipe']
#     print(region_selected, pipe_selected)
#     inputUrl = region_selected + pipe_selected
#     if not os.path.exists('../videos/output/' + region_selected + pipe_selected + '_output.mp4'):
#         processVideo(inputUrl)
#     return jsonify('Ok')


#
#
# @app.route('/live_input_video_feed')
# def live_input_video_feed():
#     print("in 1")
#     return Response(LiveProcessing.gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
# class server:
#     def __init__(self):
#         stock_in_pipe = Pipe()
#     def test(self):
#         r = request
#         # convert string of image data to uint8
#         nparr = np.fromstring(r.data, np.uint8)
#         # decode image
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#         self.stock_in_pipe.push(img)
#         # cv2.imshow("frame",img)
#
#         # plt.show()
#         # print(img.shape)
#
#         # cv2.imshow("test",img)
#         cv2.waitKey(1)
#
#         # do some fancy processing here....
#
#         # build a response dict to send back to client
#         response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
#                     }
#         # encode response using jsonpickle
#         response_pickled = jsonpickle.encode(response)
#
#         return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/person_camera_feed', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # global stock_in_pipe
    stock_in_pipe.push(img)
    # cv2.imshow("frame",img)

    # plt.show()
    print(img.shape)

    # cv2.imshow("test",img)
    # cv2.waitKey(1)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


def run_flask_server():
    app.run(host='0.0.0.0', debug=True, use_reloader=False,threaded=True)

    # cap = cv2.VideoCapture('http://192.168.31.180:8080/video')
    # while (True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()

