from flask import request,jsonify
import requests

API_ENDPOINT="http://192.168.43.38:5000/notify_zone_entry"


def post_response(zone, person):
    data = {
        'zone_id': zone,
        'person_id': person
    }
    print("value")
    r = requests.post(url=API_ENDPOINT, json=data)
    print("Entry detected :" + str(r))
    print("value")

post_response("Z1","P1")