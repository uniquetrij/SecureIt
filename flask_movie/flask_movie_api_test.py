from threading import Thread
from time import sleep

import cv2

from flask_movie.flask_movie_api import FlaskMovieAPI
from tf_session.tf_session_utils import Pipe

def gen(cap, pipe):
    i = 100

    while (i>0):
        # i-=1
        ret, image = cap.read()
        if not ret:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except:
                pass
            continue
        if not pipe.push(image):
            pipe.push_wait()
        # sleep(0.05)

if __name__ == '__main__':
    # cap0 = cv2.VideoCapture(0)
    # cap1 = cv2.VideoCapture(1)
    cap0 = cv2.VideoCapture("rtsp://admin:admin123@192.168.0.3")
    pipe0 = Pipe()
    # pipe1 = Pipe(limit=1)
    fs = FlaskMovieAPI()
    Thread(target=fs.get_app().run, args=("0.0.0.0",)).start()
    default = cv2.imread('/home/developer/Desktop/be-the-navigator.png')
    fs.create('feed_0', pipe0, default)
    # fs.create('shelf_feed', pipe1)

    Thread(target=gen, args=(cap0, pipe0,)).start()
    # Thread(target=gen, args=(cap1, pipe1,)).start()

