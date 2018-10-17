from flask import request,jsonify
from flask_cors import CORS
from flask import Flask, render_template, Response
app = Flask(__name__)
CORS(app)
@app.route('/roi_points',methods=["POST"])
def roi_points():
    global dict
    dict=request.get_json()
    print(dict)
    return "ok"


app.run(host='0.0.0.0', debug=True, use_reloader=False, threaded=True)