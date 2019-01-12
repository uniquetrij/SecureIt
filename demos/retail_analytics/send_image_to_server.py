import requests
import json
import cv2
from data.videos import path as videos_path
from flask import request

addr = 'http://localhost:5000'
test_url = addr + '/person_camera_feed'


cap = cv2.VideoCapture(videos_path.get() + '/aws_demo.mp4')

while True:
    ret, image = cap.read()
    if not ret:
        continue
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    _, img_encoded = cv2.imencode('.jpeg', image)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

    # track_persons(ret, image)

# # prepare headers for http request
# content_type = 'image/jpeg'
# headers = {'content-type': content_type}
#
# img = cv2.imread('test.jpeg')
# # encode image as jpeg
# _, img_encoded = cv2.imencode('.jpeg', img)
# # send http request with image and receive response
# response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# # decode response
# print(json.loads(response.text))
